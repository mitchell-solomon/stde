#!/usr/bin/env python
import argparse
import os
import json
import io
import logging
import pickle
from functools import partial
from typing import Callable, Tuple, Optional, get_args

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float
from jax import config

config.update("jax_enable_x64", True)
import numpy as np

from flax.training import train_state
import flax.linen as nn


import optax
from tqdm import tqdm

class TqdmToLogger(io.StringIO):
    """Redirect tqdm output to logging."""

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def count_params(params):
    flat = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(p.shape) for p in flat))


def save_params(params, step: int, save_dir: str, save_every: int, epochs: int):
    if step == 0:
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    if (step + 1) != epochs and step % save_every != 0:
        return
    with open(os.path.join(save_dir, f"params_{step}.pkl"), "wb") as f:
        pickle.dump(params, f)

import matplotlib.pyplot as plt
from pprint import pprint


from my_mamba import BidirectionalMamba, MambaConfig, DiagnosticsConfig, SSMConfig
from stde.config import EqnConfig
from stde import equations as eqns


# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PINN Training with Bi-MAMBA")

# -- existing args --
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument("--dim", type=int, default=2, help="spatial dimension of the problem")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--eval_every", type=int, default=10000000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--N_test", type=int, default=2000)
parser.add_argument("--test_batch_size", type=int, default=20)
parser.add_argument("--seq_len", type=int, default=3, help="sequence length for Bi-MAMBA")
parser.add_argument("--x_radius", type=float, default=1.0)
parser.add_argument("--x_ordering", type=str, choices=["none", "coordinate", "radial"], default="radial", help="How to order your spatial sequence: `none` (leave random), `coordinate` (sort by x[0]), `radial` (sort by ∥x∥).")

parser.add_argument(
    "--sampling_mode", type=str,
    choices=["random", "grid", "radial"],
    default="random",
    help="How to sample points: `random`=Monte‐Carlo; `grid`=even grid; "
         "`radial`=random‐start radial lines"
)

parser.add_argument(
    "--hess_diag_method",
    type=str,
    choices=[
        "stacked",
        "forward",
        "sparse_stde",
        "dense_stde",
        "scan",
        "folx",
    ],
    default="sparse_stde",
    help="method for computing the Hessian diagonal",
)
parser.add_argument(
    "--stde_dist",
    type=str,
    choices=["normal", "rademacher"],
    default="rademacher",
    help="distribution for dense STDE",
)
eqn_choices = get_args(EqnConfig.__annotations__["name"])
parser.add_argument(
    "--eqn_name",
    type=str,
    choices=eqn_choices,
    default="SineGordonTwobody",
    help="PDE to solve",
)

# numberof bidirectional mamba blocks
parser.add_argument("--num_mamba_blocks", type=int, default=1, help="number of bidirectional mamba blocks")

# -- arguments for MambaConfig --
parser.add_argument("--hidden_features",    type=int,    default=64,      help="hidden_features in each Mamba block")
parser.add_argument("--expansion_factor",   type=float,  default=2.0,     help="expansion factor in Mamba MLP")
parser.add_argument("--dt_rank",            type=str,    default="auto", choices=["auto","full","low"], help="dt_rank setting")
parser.add_argument("--activation",         type=str,    default="tanh",  choices=["silu","relu","gelu","tanh", "wave"], help="activation fn")
parser.add_argument("--norm_type",          type=str,    default="none",  choices=["none","batch","layer", "group"], help="type of normalization")
parser.add_argument("--dense_expansion",    type=int,    default=2,       help="dense expansion ratio")
parser.add_argument("--complement",         action="store_true", help="use complement flag")
parser.add_argument("--tie_in_proj",        action="store_true", help="tie input projection weights")
parser.add_argument("--tie_gate",           action="store_true", help="tie gating weights")
parser.add_argument("--bidirectional",      action="store_true", help="use bidirectional mamba")

# -- arguments for DiagnosticsConfig --
parser.add_argument("--diag_skip",     action="store_true", help="enable skip diagnostics")
parser.add_argument("--diag_gate",     action="store_true", help="enable gate diagnostics")
parser.add_argument("--diag_gated",    action="store_true", help="enable gated diagnostics")
parser.add_argument("--diag_residual", action="store_true", help="enable residual diagnostics")

# -- arguments for SSMConfig --
parser.add_argument("--recursive_scan", action="store_true", help="use recursive scan")
parser.add_argument("--min_recursion_length", type=int, default=100, help="minimum recursion length")
parser.add_argument("--recursive_split", action="store_true", help="use recursive split")
parser.add_argument("--custom_vjp_scan", action="store_true", help="use custom vjp scan")
parser.add_argument("--ssm_activation", type=str, default="silu", choices=["silu","relu","gelu","tanh", "wave"], help="activation fn for SSM")

# -- run arguments --
parser.add_argument("--run_name", type=str, default="test_run")
parser.add_argument("--save_every", type=int, default=10000,
                    help="save parameters every n steps")
parser.add_argument("--get_mem", action="store_true",
                    help="measure GPU memory usage")

args = parser.parse_args()

# derive rand_batch_size from dimension (order of magnitude lower)
rand_batch_size = max(2, args.dim // 10)
args.rand_batch_size = rand_batch_size

pprint(args)

# set up Haiku PRNG sequence for stde.operators
rng_seq = hk.PRNGSequence(args.SEED)
hk.next_rng_key = lambda: next(rng_seq)


np.random.seed(args.SEED)

# create a dir with the run name where we save all results
save_dir = f"_results/{args.run_name}"
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# logging setup
# ---------------------------------------------------------------------------
log_file = os.path.join(save_dir, "train.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file, mode="w")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())
tqdm_out = TqdmToLogger(logger)

# -----------------------------------------------------------------------------
# Domain sampler utilising equation specific sampler
# -----------------------------------------------------------------------------

sample_domain_fn = None  # placeholder, defined after equation is loaded


@partial(jax.jit, static_argnames=("batch_size", "seq_len"))
def sample_domain_seq_fn(
    batch_size: int,
    rng: jax.Array,
    seq_len: int,
) -> Tuple[jnp.ndarray, jax.Array]:
    """Sample ``batch_size``\*``seq_len`` points and reshape to sequences.

    If the equation is time dependent, the returned sequence dimension
    corresponds to the temporal axis. The sampled ``x`` and ``t`` are
    concatenated such that the model receives ``(x, t)`` as features.
    """

    if eqn.is_traj:
        x, t, _, _, rng = sample_domain_fn(batch_size, seq_len - 1, rng)
    else:
        x, t, _, _, rng = sample_domain_fn(batch_size * seq_len, 0, rng)

    # reshape to sequences
    x_seq = x.reshape((batch_size, seq_len, -1))
    if eqn.time_dependent and t is not None:
        t_seq = t.reshape((batch_size, seq_len, -1))
        # order each sequence by increasing time so that seq axis is temporal
        sort_idx = jnp.argsort(t_seq[..., 0], axis=1)
        x_seq = jnp.take_along_axis(x_seq, sort_idx[..., None], axis=1)
        t_seq = jnp.take_along_axis(t_seq, sort_idx[..., None], axis=1)
        x_seq = jnp.concatenate([x_seq, t_seq], axis=-1)
    return x_seq, rng

# -----------------------------------------------------------------------------
# Hessian‐trace estimator
# -----------------------------------------------------------------------------


coeffs_ = np.random.randn(1, args.dim)

eqn_cfg = EqnConfig(
    name=args.eqn_name,
    dim=args.dim,
    max_radius=args.x_radius,
    rand_batch_size=rand_batch_size,
    hess_diag_method=args.hess_diag_method,
    stde_dist=args.stde_dist,
)

eqn = getattr(eqns, eqn_cfg.name)
if eqn.random_coeff:
    eqn_cfg.coeffs = coeffs_
else:
    eqn_cfg.coeffs = None

# sampler for boundary points using equation-specific sampling
sample_domain_fn = eqn.get_sample_domain_fn(eqn_cfg)
sample_boundary_fn = sample_domain_fn

if eqn.time_dependent:
    sol_fn = lambda xt: eqn.sol(xt[..., :args.dim], xt[..., args.dim:], eqn_cfg)

    def residual_fn(xt, u_fn: Callable) -> Float[Array, "xt_dim"]:
        x_part = xt[..., :args.dim]
        t_part = xt[..., args.dim:]
        res = eqn.res(
            x_part,
            t_part,
            lambda xi, ti: u_fn(jnp.concatenate([xi, ti], axis=-1)),
            eqn_cfg,
        )
        if isinstance(res, tuple):
            res = res[0]
        return res
else:
    sol_fn = lambda x: eqn.sol(x, None, eqn_cfg)

    def residual_fn(x, u_fn: Callable) -> Float[Array, "xt_dim"]:
        res = eqn.res(x, None, lambda xi, _t: u_fn(xi), eqn_cfg)
        if isinstance(res, tuple):
            res = res[0]
        return res

# -----------------------------------------------------------------------------
# Training state
# -----------------------------------------------------------------------------
class MambaTrainState(train_state.TrainState):
    rng: jax.Array


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # rng setup
    master_rng = jax.random.PRNGKey(args.SEED)
    rng_train, rng_test = jax.random.split(master_rng)

    # instantiate Bi-MAMBA
    mamba_cfg = MambaConfig(
        hidden_features       = args.hidden_features,
        expansion_factor      = args.expansion_factor,
        dt_rank               = args.dt_rank,
        activation            = args.activation,
        norm_type             = args.norm_type,
        dense_expansion       = args.dense_expansion,
        complement            = args.complement,
        tie_in_proj           = args.tie_in_proj,
        tie_gate              = args.tie_gate,
        radius                = args.x_radius,
        bidirectional         = args.bidirectional,
        diagnostics           = DiagnosticsConfig(
            skip     = args.diag_skip,
            gate     = args.diag_gate,
            gated    = args.diag_gated,
            residual = args.diag_residual,
        ),
    )

    ssm_cfg = SSMConfig(
        recursive_scan      = args.recursive_scan,
        min_recursion_length   = args.min_recursion_length,
        recursive_split     = args.recursive_split,
        custom_vjp_scan     = args.custom_vjp_scan,
        activation           = args.ssm_activation,
    )
    # make a class for the PINN, which is a stack of Bi-MAMBA blocks
    class BiMambaPINN(nn.Module):
        eqn: eqns.Equation
        eqn_cfg: EqnConfig
        @nn.compact
        def __call__(self, x):
            # Ensure input shape is (B, L, D)
            B = x.shape[0]
            D = x.shape[-1]
            L = x.shape[-2]
            x_in = x

            # Apply the Mamba model
            for i in range(args.num_mamba_blocks):
                x = BidirectionalMamba(**vars(mamba_cfg), ssm_args=vars(ssm_cfg))(x)
            
            x_out = nn.Dense(args.dense_expansion*D, name="mlp", kernel_init=nn.initializers.lecun_normal())(x)
            x_out = nn.gelu(x_out)
            x_out = nn.Dense(1, name="mlp_proj", kernel_init=nn.initializers.lecun_normal())(x_out)  # (B, L, D) --> (B, L, 1)
            x_out = x_out.squeeze(-1)
            # enforce PDE-specific boundary condition
            if self.eqn.time_dependent:
                x_part, t_part = x_in[..., :args.dim], x_in[..., args.dim:]
                x_out = self.eqn.enforce_boundary(x_part, t_part, x_out, self.eqn_cfg)
            else:
                x_out = self.eqn.enforce_boundary(x_in, None, x_out, self.eqn_cfg)

            return x_out

        def tabulate_model(self,
                           n_pts: int = 4,
                           dim: int = args.dim,
                           radius: float = args.x_radius,
                           seq_len: int = args.seq_len,
                           rng: Optional[jax.Array] = None):
            """Print a tabulated summary of the model architecture."""

            if rng is None:
                rng = jax.random.PRNGKey(0)

            x, _ = sample_domain_seq_fn(
                batch_size=n_pts,
                rng=rng,
                seq_len=seq_len,
            )

            print(nn.tabulate(
                self,
                rngs={"params": rng},
                mutable=["params", "diagnostics", "intermediates"],
            )(x))

    # And then proceed to instantiate your model as before:
    mamba = BiMambaPINN(eqn=eqn, eqn_cfg=eqn_cfg)

    # initialize parameters on a dummy sequence
    rng_train, init_rng = jax.random.split(rng_train)
    x_dummy, rng = sample_domain_seq_fn(
        batch_size=2,
        rng=init_rng,
        seq_len=args.seq_len,
    )
    flax_vars = mamba.init(init_rng, x_dummy)

    # print input and output shapes
    print("Input shape:", x_dummy.shape)
    print("Output shape:", mamba.apply(flax_vars, x_dummy).shape)

    # Tabulate the model architecture for reference
    mamba.tabulate_model()
    
    # init exponential decay learning rate schedule
    
    lr = args.lr
    # lr = optax.exponential_decay(
    #     init_value=args.lr,
    #     transition_steps=args.epochs,
    #     transition_begin=args.epochs//3,
    #     decay_rate=0.0001
    # )
    
    # # plot lr vs epoch
    # epochs = list(range(args.epochs))
    # lrs = [lr(epoch) for epoch in epochs]
    # plt.plot(epochs, lrs)
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.yscale('log')
    # plt.title('Learning Rate Schedule')
    # plt.show()


    # create optimizer + train state
    optimizer = optax.adam(learning_rate=lr)
    state = MambaTrainState.create(
        apply_fn=mamba.apply,
        params=flax_vars["params"],
        tx=optimizer,
        rng=rng_train,
    )
    num_params = count_params(state.params)
    logger.info(f"num params: {num_params}")
    gpu_mems = [0.0]

    # prepare test set (once)
    test_seqs = test_truths = None
    if not eqn.is_traj:
        n_test_batches = args.N_test // args.test_batch_size
        test_seqs = []
        test_truths = []

        for _ in range(n_test_batches):
            rng_test, sample_rng = jax.random.split(rng_test)
            x_test_seq, _ = sample_domain_seq_fn(
                batch_size=args.test_batch_size,
                rng=sample_rng,
                seq_len=args.seq_len,
            )
            # collapse batch & seq dims to analytical solver
            B, L, D = x_test_seq.shape
            x_flat = x_test_seq.reshape((B * L, D))
            y_flat = jax.vmap(sol_fn)(x_flat)
            test_seqs.append(x_test_seq)
            test_truths.append(y_flat.reshape((B, L)))
        test_seqs = jnp.stack(test_seqs)       # (n_batches, B, L, D)
        test_truths = jnp.stack(test_truths)   # (n_batches, B, L)

        # flatten all test ground-truth values into a single vector
        y_true_all = test_truths.reshape(-1)

        # L1 norm of the entire test set
        y_true_l1 = float(jnp.sum(jnp.abs(y_true_all)))

        # L2 norm of the entire test set
        y_true_l2 = float(jnp.linalg.norm(y_true_all))
    else:
        # reference value only defined at x=0,t=0
        y_ref = eqn.sol(jnp.zeros((args.dim,)), jnp.zeros((1,)), eqn_cfg)
        y_true_l1 = jnp.abs(y_ref)
        y_true_l2 = jnp.abs(y_ref)

    @jax.jit
    def train_step(state: MambaTrainState) -> MambaTrainState:
        # 1) split off one rng for sampling, one to carry forward
        batch_rng, next_rng = jax.random.split(state.rng)
        x_seq, batch_rng = sample_domain_seq_fn(
            batch_size=rand_batch_size,
            rng=batch_rng,
            seq_len=args.seq_len,
        )  # x_seq: (B, L, D)

        def loss_fn(params, rng):
            """
            Computes the mean-squared PDE residual loss for a batch given parameters.

            Returns:
            loss:       scalar MSE of all residuals
            new_rng:    PRNGKey to carry forward
            """
            # 1) split RNG for sampling vs. carry-forward
            batch_rng, new_rng = jax.random.split(rng)

            # 2) sample a fresh batch of input sequences
            x_seq, batch_rng = sample_domain_seq_fn(
                batch_size=rand_batch_size,
                rng=batch_rng,
                seq_len=args.seq_len,
            )  # x_seq shape: (B, L, D)

            # 3) helper: model output at time index l
            def y_at_l(xt_i, l, full_seq):
                # replace the l-th entry of full_seq with xt_i
                seq2 = lax.dynamic_update_slice(full_seq, xt_i[None, :], (l, 0))
                # apply your Bi-MAMBA with the passed-in params
                y2 = mamba.apply({"params": params}, seq2[None, ...]).squeeze(0)
                return y2[l]

            # 4) compute the vector of residuals for one sequence

            def residuals_for_one_sequence(full_seq, key):
                L = full_seq.shape[0]

                def one_step_res(l, xt_l, key):
                    # build a u_fn that closes over `params` and this full_seq
                    def u_fn(xt_i):
                        return y_at_l(xt_i, l, full_seq)

                    res_val = residual_fn(xt_l, u_fn)
                    return res_val, key

                # split into L subkeys, run across time steps
                keys = jax.random.split(key, L)
                resids, _ = jax.vmap(one_step_res, in_axes=(0, 0, 0), out_axes=(0, 0))(
                    jnp.arange(L), full_seq, keys
                )
                return resids, _

            # 5) vectorize over the B sequences in the batch
            outer_keys = jax.random.split(batch_rng, x_seq.shape[0])
            all_resids, _ = jax.vmap(
                residuals_for_one_sequence, in_axes=(0, 0), out_axes=(0, 0)
            )(x_seq, outer_keys)  # all_resids shape: (B, L)

            # 6) mean-squared residual loss
            domain_loss = jnp.mean(all_resids ** 2)
            _, _, x_b, t_b, batch_rng = sample_boundary_fn(
                rand_batch_size, rand_batch_size, batch_rng
            )
            if eqn.time_dependent:
                xt_b = jnp.concatenate([x_b, t_b], axis=-1)
                u_b = mamba.apply({"params": params}, xt_b[:, None, :]).squeeze()
            else:
                u_b = mamba.apply({"params": params}, x_b[:, None, :]).squeeze()
            g_b = eqn.boundary_cond(x_b, t_b, eqn_cfg)
            g_b = eqn.boundary_cond(x_b, t_b, eqn_cfg)
            boundary_loss = jnp.mean((g_b - u_b) ** 2)

            loss = (
                eqn_cfg.domain_weight * domain_loss
                + eqn_cfg.boundary_weight * boundary_loss
            )
            new_rng = batch_rng
            return loss, new_rng

        # inside your jitted train_step
        (loss, new_rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state.rng
        )
        state = state.apply_gradients(grads=grads, rng=new_rng)

        return state, loss, grads
    
    losses = []
    iters = tqdm(range(args.epochs), file=tqdm_out)
    for step in iters:
        state, train_loss, grads = train_step(state)
        losses.append(float(train_loss))

        if step % args.eval_every == 0:
            if not eqn.is_traj:
                l1_total, l2_total_sqr = 0.0, 0.0
                for b in range(test_seqs.shape[0]):
                    x_seq = test_seqs[b]
                    y_true = test_truths[b]
                    y_pred = mamba.apply({"params": state.params}, x_seq)
                    err = y_pred - y_true
                    l1_total += jnp.sum(jnp.abs(err))
                    l2_total_sqr += jnp.sum(err**2)
                l1_rel = float(l1_total / y_true_l1)
                l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)
            else:
                xt_zero = jnp.zeros((1, args.seq_len, args.dim + 1))
                y_pred = mamba.apply({"params": state.params}, xt_zero)[0, 0]
                y_true = eqn.sol(jnp.zeros((args.dim,)), jnp.zeros((1,)), eqn_cfg)
                l1_rel = float(jnp.abs(y_pred - y_true) / jnp.abs(y_true))
                l2_rel = l1_rel

            grad_norm = float(jnp.sqrt(
                sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
            ))

            desc_str = (
                f"l1_rel={l1_rel:.2e} | l2_rel={l2_rel:.2e} | "
                f"loss={train_loss:.2e} | grad_norm={grad_norm:.2e}"
            )
            iters.set_description(desc_str)
            logger.info(desc_str)

            if args.get_mem and step == 100:
                mem_stats = jax.local_devices()[0].memory_stats()
                peak_mem = mem_stats['peak_bytes_in_use'] / 1024**2
                gpu_mems.append(peak_mem)
                break

        if step % args.save_every == 0:
            save_params(state.params, step, save_dir, args.save_every, args.epochs)

    # read iter/s from log file
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        iter_per_s = float(lines[-3].strip().split(', ')[-1].split('it/s')[0])
    except Exception:
        iter_per_s = 0.0

    # --- Plot training loss curve ---
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f"{save_dir}/training_loss_curve.png", dpi=300)
    # plt.show()

    # --- Final evaluation on the full test set ---
    print("\n=== Final evaluation on test set ===")
    if not eqn.is_traj:
        l1_total, l2_total_sqr = 0.0, 0.0
        for b in range(test_seqs.shape[0]):
            x_seq = test_seqs[b]               # (B, L, D)
            y_true = test_truths[b]            # (B, L)

            y_pred = mamba.apply({"params": state.params}, x_seq)
            # drop trailing dim if present
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)

            err = y_pred - y_true
            l1_total       += jnp.sum(jnp.abs(err))
            l2_total_sqr   += jnp.sum(err**2)

        l1_rel = float(l1_total / y_true_l1)
        l2_rel = float(jnp.sqrt(l2_total_sqr) / y_true_l2)
        print(f"Final → l1_rel={l1_rel:.3e} | l2_rel={l2_rel:.3e}")
    else:
        xt_zero = jnp.zeros((1, args.seq_len, args.dim + 1))
        y_pred = mamba.apply({"params": state.params}, xt_zero)[0, 0]
        y_true = eqn.sol(jnp.zeros((args.dim,)), jnp.zeros((1,)), eqn_cfg)
        l1_rel = float(jnp.abs(y_pred - y_true) / jnp.abs(y_true))
        l2_rel = l1_rel
        print(f"Final → l1_rel={l1_rel:.3e}")

    with open(f"{save_dir}/final_eval_results.json", "w") as f:
        json.dump({
            "l1_rel": l1_rel,
            "l2_rel": l2_rel,
            "iter_per_s": iter_per_s,
            "peak_gpu_mem": max(gpu_mems),
            "num_params": num_params,
            "final_loss": float(losses[-1]) if losses else 0.0
        }, f, indent=2)


    # --- Plotting ---


    def _title_for_eqn(name: str, cfg: EqnConfig) -> str:
        params = []
        if name == "HJB_LQG":
            params.append(f"mu={cfg.mu}")
        if name == "HJB_LIN":
            params.append(f"c={cfg.c}")
        if name == "BSB":
            params.append(f"sigma={cfg.sigma}")
            params.append(f"r={cfg.r}")
        if hasattr(cfg, "max_radius"):
            params.append(f"max_r={cfg.max_radius}")
        parts = ", ".join(params)
        bc = eqn.boundary_cond.__name__
        return f"{name} ({parts})\nBC: {bc}"

    def plot_solution(x_flat, u_true, u_pred,
                      dim: int | None = None,
                      cmap: str = 'viridis',
                      eqn_name: str = args.eqn_name):
        """Plot true, predicted and diff solution with PDE meta info."""

        diff = u_pred - u_true

        if eqn.time_dependent or eqn.is_traj:
            xi = np.array(x_flat[:, 0])
            yi = np.array(x_flat[:, -1])
            xlabel, ylabel = 'x', 't'

            vmin = min(np.min(u_true), np.min(u_pred))
            vmax = max(np.max(u_true), np.max(u_pred))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            sc0 = axes[0].scatter(xi, yi, c=u_true, cmap=cmap,
                                  vmin=vmin, vmax=vmax, s=20)
            axes[0].set_title('True u')
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel(ylabel)

            sc1 = axes[1].scatter(xi, yi, c=u_pred, cmap=cmap,
                                  vmin=vmin, vmax=vmax, s=20)
            axes[1].set_title('Predicted u')
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)

            sc2 = axes[2].scatter(xi, yi, c=diff, cmap='coolwarm', s=20)
            axes[2].set_title('Difference')
            axes[2].set_xlabel(xlabel)
            axes[2].set_ylabel(ylabel)

            fig.colorbar(sc0, ax=axes[:2], shrink=0.8, pad=0.02, label='u')
            fig.colorbar(sc2, ax=axes[2], shrink=0.8, pad=0.02, label='Δu')

        else:
            if dim is None:
                D = x_flat.shape[1] if x_flat.ndim > 1 else 1
            else:
                D = dim

            if D > 2:
                print(f"Warning: embedding dim={D} > 2; plotting only first two dims.")
                D = 2

            if D == 1:
                x = x_flat if x_flat.ndim == 1 else x_flat[:, 0]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].plot(x, u_true, '.', label='true')
                axes[0].set_title('True u(x)')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('u')

                axes[1].plot(x, u_pred, '.', color='C1')
                axes[1].set_title('Predicted u(x)')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('u')

                axes[2].plot(x, diff, '.', color='C2')
                axes[2].set_title('Difference')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('Δu')

            elif D == 2:
                xi = x_flat[:, 0]
                yi = x_flat[:, 1]

                vmin = min(np.min(u_true), np.min(u_pred))
                vmax = max(np.max(u_true), np.max(u_pred))

                fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                       subplot_kw={'aspect': 'equal'})

                sc0 = axes[0].scatter(xi, yi, c=u_true, cmap=cmap,
                                      vmin=vmin, vmax=vmax, s=20)
                axes[0].set_title('True u')
                axes[0].set_xlabel('dim0')
                axes[0].set_ylabel('dim1')

                sc1 = axes[1].scatter(xi, yi, c=u_pred, cmap=cmap,
                                      vmin=vmin, vmax=vmax, s=20)
                axes[1].set_title('Predicted u')
                axes[1].set_xlabel('dim0')
                axes[1].set_ylabel('dim1')

                sc2 = axes[2].scatter(xi, yi, c=diff, cmap='coolwarm', s=20)
                axes[2].set_title('Difference')
                axes[2].set_xlabel('dim0')
                axes[2].set_ylabel('dim1')

                fig.colorbar(sc0, ax=axes[:2], shrink=0.8, pad=0.02, label='u')
                fig.colorbar(sc2, ax=axes[2], shrink=0.8, pad=0.02, label='Δu')

            else:
                raise ValueError('plot only supports 1D or 2D embeddings for visualization')

        fig.suptitle(_title_for_eqn(eqn_name, eqn_cfg))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{eqn_name}_solution.png", dpi=300)

    def plot_all_solutions(test_seqs, test_truths, params, mamba, dim=None,
                           xlabel='x', ylabel='u', cmap='viridis',
                           eqn_name: str = args.eqn_name):
        """
        Run the model over every test batch, flatten, and plot true vs. predicted.
        
        Args:
        test_seqs:   array (n_batches, B, L, D)
        test_truths: array (n_batches, B, L)
        params:      your trained params to pass into mamba.apply
        mamba:       your BidirectionalMamba instance
        dim:         spatial dimension (if None, inferred from D)
        """
        # flatten batches
        n_batches, B, L, D = test_seqs.shape
        if dim is None:
            dim = D

        # reshape to (n_batches*B, L, D) for a big batch
        seqs_flat = test_seqs.reshape((n_batches * B, L, D))
        # run model in one go
        u_pred_seq = mamba.apply({"params": params},
                                seqs_flat)      # → (n_batches*B, L, 1) or (nB, L)
        
        # flatten spatial points and truths
        x_flat = test_seqs.reshape((n_batches * B * L, D))
        u_true = test_truths.reshape((n_batches * B * L,))
        u_pred = u_pred_seq.reshape((n_batches * B * L,))

        # convert to NumPy for plotting
        x_flat_np = np.array(x_flat)
        u_true_np  = np.array(u_true)
        u_pred_np  = np.array(u_pred)

        # now call the existing plot fn
        plot_solution(x_flat_np,
                      u_true_np,
                      u_pred_np,
                      dim=dim,
                      cmap=cmap,
                      eqn_name=eqn_name)

    if not eqn.is_traj:
        plot_all_solutions(test_seqs,
                           test_truths,
                           state.params,
                           mamba,
                           dim=args.dim)
    def visualize_sequences(x_seq, x_ordering="none"):
        """
        Scatter‐plot each 2D “sequence” in x_seq, coloring by sequence index.
        If x_ordering="radius" or "coordinate", also draw lines to show the ordering.

        Args:
        x_seq:       numpy array of shape (B, L, 2)
        x_ordering:  "none", "coordinate", or "radial"
        """
        B, L, D = x_seq.shape
        assert D == 2, "visualize_sequences only supports 2D inputs"

        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(6,6))

        for b in range(B):
            pts = x_seq[b]  # (L, 2)

            # apply ordering if requested
            if x_ordering == "coordinate":
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
            elif x_ordering == "radial":
                idx = np.argsort(np.linalg.norm(pts, axis=1))
                pts = pts[idx]
            # else "none": leave as is

            color = cmap(b % 10)
            plt.scatter(pts[:,0], pts[:,1], color=color, s=30, label=f"seq {b}")

            # if ordered, draw lines to reveal the path
            if x_ordering in ("coordinate", "radial"):
                plt.plot(pts[:,0], pts[:,1], color=color, alpha=0.5)

        plt.title(f"Sequences on 2D ball (ordering={x_ordering})")
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("x₀")
        plt.ylabel("x₁")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sequences_{x_ordering}.png", dpi=300)
        # plt.show()
    if not eqn.is_traj:
        seq_vis = test_seqs[0]
        if seq_vis.shape[-1] > 2:
            seq_vis = seq_vis[..., :2]
        visualize_sequences(seq_vis, x_ordering=args.x_ordering)



if __name__ == "__main__":
    main()
