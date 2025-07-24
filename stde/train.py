import io
import logging
import time
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import optax
import wandb
from absl import logging as absl_log
from tqdm import tqdm

import stde.equations as eqns
from stde.config import Config
from stde.model import PINN
from stde.optimize import get_optimizer
from stde.types import Equation, TrainingState


class TqdmToLogger(io.StringIO):
  """Output stream for TQDM which will output to logger module instead of the StdOut.
  """
  logger = None
  level = None
  buf = ''

  def __init__(self, logger, level=None):
    super(TqdmToLogger, self).__init__()
    self.logger = logger
    self.level = level or logging.INFO

  def write(self, buf):
    self.buf = buf.strip('\r\n\t ')

  def flush(self):
    self.logger.log(self.level, self.buf)


def count_params(params):
  flat_params = jax.tree_leaves(params)
  return sum([np.prod(p.shape) for p in flat_params])


def train(cfg: Config, use_wandb: bool, run_id: int = 1, get_mem: bool = False):
  if use_wandb:
    name = cfg.get_run_name()
    wandb.init(
      name=name + str(run_id),
      project="jax-pinn",
      config=cfg,
      group=name,
      job_type='train'
    )

  rng = jax.random.PRNGKey(cfg.rng_seed)

  # get equation
  eqn: Equation = getattr(eqns, cfg.eqn_cfg.name)

  sample_domain_fn = eqn.get_sample_domain_fn(cfg.eqn_cfg)

  # prepare dummy data for init
  x, t, _, _, rng = sample_domain_fn(2, 2, rng)

  # init model
  model = hk.multi_transform(
    lambda: PINN(
      eqn=eqn,
      eqn_cfg=cfg.eqn_cfg,
      model_cfg=cfg.model_cfg,
    ).init_for_multitransform()
  )
  key, rng = jax.random.split(rng)
  params = model.init(key, x, t)

  num_params = count_params(params)
  logging.info(f"num params: {num_params}")

  if use_wandb:
    wandb.log(dict(num_params=num_params))

  # prepare test data
  if not eqn.is_traj and eqn.offline_sol == "":
    y_t_true_l2 = 1.  # placeholder
    np_ = jnp if cfg.test_cfg.data_on_gpu else np
    n_test_batches = cfg.test_cfg.n_points // cfg.test_cfg.batch_size
    assert n_test_batches > 0
    x_tests, t_tests, y_trues, y_t_trues = [], [], [], []
    for _ in tqdm(range(n_test_batches), desc="generating test data..."):
      # NOTE: why not test boundary condition?
      x_test_i, t_test_i, _, _, rng = sample_domain_fn(
        cfg.test_cfg.batch_size, 1, rng
      )
      y_true_i = eqn.sol(x_test_i, t_test_i, cfg.eqn_cfg)
      x_tests.append(np_.array(x_test_i))
      t_tests.append(np_.array(t_test_i))
      y_trues.append(np_.array(y_true_i))

      if eqn.time_dependent and cfg.model_cfg.compute_w1_loss:
        y_t_true_i = jax.vmap(
          jax.grad(partial(eqn.sol, cfg=cfg.eqn_cfg), argnums=1)
        )(x_test_i, t_test_i)
        y_t_trues.append(np_.array(y_t_true_i))

    y_true = np_.hstack(y_trues)
    if eqn.time_dependent and cfg.model_cfg.compute_w1_loss:
      y_t_true = np_.hstack(y_t_trues)
      y_t_true_l2 = np_.linalg.norm(y_t_true)

    y_true_l1, y_true_l2 = [np_.linalg.norm(y_true, ord=ord) for ord in [1, 2]]
    y_true_linf = float(np_.max(np_.abs(y_true)))

  elif eqn.offline_sol != "":
    y_true_linf = 1.

  @jax.jit
  def update(state: TrainingState) -> Tuple:
    """sample from domain then update parameter"""
    rng = state.rng_key
    if eqn.is_traj:
      x, t, x_boundary, t_boundary, rng = sample_domain_fn(
        cfg.eqn_cfg.n_traj, cfg.eqn_cfg.n_t, rng
      )
    else:
      x, t, x_boundary, t_boundary, rng = sample_domain_fn(
        cfg.eqn_cfg.batch_size, cfg.eqn_cfg.batch_size_boundary, rng
      )
    key, rng = jax.random.split(rng)
    if cfg.gd_cfg.n_fgd_vec != 0:
      loss_fn_ = lambda p_: model.apply.loss_fn(
        p_, key, x, t, x_boundary, t_boundary
      )
      (loss, aux), grad_fn = jax.linearize(loss_fn_, state.params)

      # TODO: use new key for each rand vec
      key2, rng = jax.random.split(rng)
      rand_vec = jax.tree_util.tree_map(
        lambda x: jax.random.normal(
          key2,
          shape=(cfg.gd_cfg.n_fgd_vec, *x.shape),
        ), state.params
      )
      c_loss, _ = jax.vmap(grad_fn)(rand_vec)
      grad = jax.tree_util.tree_map(
        lambda v: (c_loss.reshape(-1, *([1] * (len(v.shape) - 1))) * v).mean(0),
        rand_vec
      )

    else:
      val_and_grads_fn = jax.value_and_grad(model.apply.loss_fn, has_aux=True)
      (loss, aux), grad = val_and_grads_fn(
        state.params, key, x, t, x_boundary, t_boundary
      )
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(params, opt_state, rng), aux

  # init optimizers
  opt_states = get_optimizer(cfg.gd_cfg, params, rng)
  optimizer, state = opt_states["main"]

  err_norms_jit = jax.jit(model.apply.err_norms_fn)

  losses = []
  l1_rels, l2_rels, w1_t_rels = [], [], []
  linf_rels, res_means, res_maxs = [], [], []
  l1_rel = l2_rel = w1_t_rel = 0.

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  tqdm_out = TqdmToLogger(logger, level=logging.INFO)

  iters = tqdm(range(cfg.gd_cfg.epochs), file=tqdm_out)

  gpu_mems = [0.]
  epoch_times = []
  start_time = time.time()
  calc_w1 = eqn.time_dependent and cfg.model_cfg.compute_w1_loss
  for step in iters:

    step_start = time.time()

    loss, state, aux = update(state)

    losses.append(loss)

    if use_wandb and step % cfg.test_cfg.log_every == 0:
      wandb.log({**dict(loss=loss), **aux}, step=step)

    if (step + 1) % cfg.test_cfg.eval_every == 0 or (get_mem and step == 100):

      if not eqn.is_traj:
        l1_total, l2_total_sqr, linf_val = 0., 0., 0.
        res_sum, res_max = 0., 0.
        w1_t_total_sqr = 0.
        for i in range(n_test_batches):
          l1, l2_sqr, w1_t_sqr = err_norms_jit(
            state.params, state.rng_key, x_tests[i], t_tests[i], y_trues[i],
            y_t_trues[i] if calc_w1 else None
          )
          l1_total += l1
          l2_total_sqr += l2_sqr
          w1_t_total_sqr += w1_t_sqr

          y_pred = jax.vmap(lambda x_, t_: model.apply.u(state.params, state.rng_key, x_, t_))(x_tests[i], t_tests[i])
          err = jnp.abs(y_trues[i] - y_pred)
          linf_val = max(linf_val, float(jnp.max(err)))
          res_batch = jax.vmap(lambda x_, t_: eqn.res(x_, t_, lambda x1, t1: model.apply.u(state.params, state.rng_key, x1, t1), cfg.eqn_cfg))(x_tests[i], t_tests[i])
          res_sum += float(jnp.mean(jnp.abs(res_batch)))
          res_max = max(res_max, float(jnp.max(jnp.abs(res_batch))))

        l1_rel = float(l1_total / y_true_l1)
        l2_rel = float(l2_total_sqr**0.5 / y_true_l2)
        linf_rel = float(linf_val / y_true_linf)
        res_mean = float(res_sum / n_test_batches)
        res_max_val = float(res_max)
        w1_t_rel = float(w1_t_total_sqr**0.5 / y_t_true_l2)

        l1_rels.append(l1_rel)
        l2_rels.append(l2_rel)
        w1_t_rels.append(w1_t_rel)
        linf_rels.append(linf_rel)
        res_means.append(res_mean)
        res_maxs.append(res_max_val)

        desc_str = f"{l1_rel=:.2E} | {l2_rel=:.2E} | {loss=:.2E} | "
        desc_str += " | ".join(
          [f"{k}={v:.2E}" for k, v in aux.items() if v != 0.0]
        )
      else:
        # TODO: fix these harded values
        # test point at x = (0, 0, ...), t = 0
        x = jnp.zeros((cfg.eqn_cfg.dim,))
        t = jnp.zeros((1,))
        y_ref = eqn.sol(x, t, cfg.eqn_cfg)
        y_pred = model.apply.u(state.params, state.rng_key, x, t)
        l1_rel = jnp.abs(y_ref - y_pred) / jnp.abs(y_ref)

        l1_rels.append(l1_rel)
        l2_rels.append(l1_rel)

        desc_str = f"{l1_rel=:.2E} | {loss=:.2E} | "
        desc_str += " | ".join(
          [f"{k}={v:.2E}" for k, v in aux.items() if v != 0.0]
        )

      if cfg.test_cfg.show_stats:
        iters.set_description(desc_str)
      else:
        logging.info(desc_str)

      if use_wandb:
        wandb.log(
          dict(l1_rel=l1_rel, l2_rel=l2_rel, linf_rel=linf_rel,
               res_mean=res_mean, res_max=res_max_val,
               w1_t_rel=w1_t_rel), step=step
        )

      if get_mem:
        mem_stats = jax.local_devices()[0].memory_stats()
        peak_mem = mem_stats['peak_bytes_in_use'] / 1024**2
        print({k: v / 1024**2 for k, v in mem_stats.items()})
        print(f"peak mem: {peak_mem:5.2f}MBs")  #
        gpu_mems.append(peak_mem)
        break

    if step % cfg.test_cfg.save_every == 0:
      cfg.save_params(state.params, step)

    epoch_times.append(time.time() - step_start)

  with open(absl_log.get_log_file_name(), 'r') as f:
    lines = f.readlines()

  if use_wandb:
    wandb.finish()

  log_path = absl_log.get_log_file_name()
  with open(log_path, 'r') as f:
    lines = f.readlines()

  if get_mem:
    iter_per_s = 0.

  else:
    try:
      iter_per_s = float(lines[-3].strip().split(', ')[-1].split('it/s')[0])
    except Exception as e:
      logging.warn(e)
      iter_per_s = float(lines[-4].strip().split(', ')[-1].split('it/s')[0])

  total_time = time.time() - start_time
  time_per_epoch = sum(epoch_times) / len(epoch_times)

  mass_err = energy_err = 0.0
  if cfg.eqn_cfg.name == "KdV":
    sample_x, sample_t, _, _, rng = sample_domain_fn(cfg.eqn_cfg.mc_batch_size, 0, rng)
    t0 = jnp.zeros_like(sample_t)
    tT = jnp.ones_like(sample_t) * cfg.eqn_cfg.T
    u_fn = lambda x, t: model.apply.u(state.params, state.rng_key, x, t)
    mass0 = eqns.KdV_mass(sample_x, t0, u_fn, cfg.eqn_cfg)
    massT = eqns.KdV_mass(sample_x, tT, u_fn, cfg.eqn_cfg)
    energy0 = eqns.KdV_energy(sample_x, t0, u_fn, cfg.eqn_cfg)
    energyT = eqns.KdV_energy(sample_x, tT, u_fn, cfg.eqn_cfg)
    mass_err = float(jnp.abs(massT - mass0))
    energy_err = float(jnp.abs(energyT - energy0))

  return dict(
      losses=losses,
      l1_rels=l1_rels,
      l2_rels=l2_rels,
      linf_rels=linf_rels,
      res_means=res_means,
      res_maxs=res_maxs,
      iter_per_s=iter_per_s,
      time_per_epoch=time_per_epoch,
      total_time=total_time,
      peak_gpu_mem=max(gpu_mems),
      mass_error=mass_err,
      energy_error=energy_err,
  )
