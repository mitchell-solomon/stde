from typing import Callable, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from folx import forward_laplacian
from jax import lax
from jax.experimental import jet
from jaxtyping import Array, Float, Integer

import stde.types as types
from stde import types
from stde.config import EqnConfig


def partial_i(fn: Callable, i: int, *args):
  """Returns a partial function where all except the i-th argument are fixed."""
  args = list(args)
  p_args = args[:i] + [None] + args[i + 1:]
  return lambda x: fn(*[arg if arg is not None else x for arg in p_args])


def get_sdgd_idx_set(cfg: EqnConfig) -> Array:
  if cfg.rand_batch_size != 0:
    key = hk.next_rng_key()
    idx_set = jax.random.choice(
      key, cfg.dim, shape=(cfg.rand_batch_size,), replace=False
    )
  else:
    idx_set = jnp.arange(cfg.dim)
  return idx_set


def hvp(f, x, v):
  """reverse-mode Hessian-vector product"""
  return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)


def get_hutchinson_random_vec(
  idx_set: Array,
  cfg: EqnConfig,
  with_time: bool = False,
) -> Float[Array, "x_dim"]:
  """Return a random vector on sdgd sampled dimensions, with Rademacher
  distribution.

  If with_time, add a one-hot time dimension at the end.
  """
  key = hk.next_rng_key()
  d = cfg.rand_batch_size or cfg.dim
  if cfg.stde_dist == "normal":
    rand_vec = jax.random.normal(key, shape=(cfg.rand_batch_size, d))
  elif cfg.stde_dist == "rademacher":
    rand_vec = 2 * (
      jax.random
      .randint(key, shape=(cfg.rand_batch_size, d), minval=0, maxval=2) - 0.5
    )
  else:
    raise ValueError
  n_vec = cfg.rand_batch_size if not with_time else cfg.rand_batch_size + 1
  d = cfg.dim if not with_time else cfg.dim + 1
  rand_sub_vec = jnp.zeros((n_vec, d)).at[:cfg.rand_batch_size,
                                          idx_set].set(rand_vec)
  if with_time:
    rand_sub_vec = rand_sub_vec.at[-1].set(jnp.eye(d)[-1])
  return rand_sub_vec


def hte(
  fn: Callable,
  cfg: EqnConfig,
  argnums: int = 0,
) -> Callable:

  def fn_trace(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "1"],
  ]:
    x_i = xs[argnums]
    dim = cfg.dim

    f_partial = partial_i(fn, argnums, *xs)
    idx_set = get_sdgd_idx_set(cfg)

    rand_sub_vec = get_hutchinson_random_vec(idx_set, cfg)

    taylor_2 = lambda v: jet.jet(
      fun=f_partial,
      primals=(x_i,),
      series=((v, jnp.zeros(dim)),),
    )

    f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_sub_vec)
    trace_est = jnp.mean(hvps)

    return idx_set, f_vals[0], trace_est

  return fn_trace


def hess_diag(
  fn: Callable,
  cfg: EqnConfig,
  argnums: int = 0,
  with_time: bool = False,
) -> Callable:
  """Given a multi-input multivariate scalar function f(x_1, ..., x_n) where
  each x_i can be multi-dimensional, returns a new vector function that computes
  the hessian diagonal of f w.r.t. to the i-th input x_i.

  Also returns the function evaluation, and the first order gradient
  :math:`\partial_i f` as well. Note that the function evaluation is not
  affected by idx_set, and the first order gradient is not sampled unless
  rand_jac is True.

  Args:
    with_time: if true, the last dimension of x_i is time, which will not be
      sampled.

  Returns:
    idx_set: a subset of dimension indices of x_i. The laplacian is only
      computed w.r.t. these dimensions.
  """

  def fn_hess_diag(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "xi_dim"],
    Float[Array, "xi_dim"],
  ]:
    x_i = xs[argnums]
    dim = cfg.dim if not with_time else cfg.dim + 1

    f_partial = partial_i(fn, argnums, *xs)
    idx_set = get_sdgd_idx_set(cfg)

    if cfg.hess_diag_method == "folx":
      assert not with_time

      fwd_f = forward_laplacian(f_partial)
      result = fwd_f(x_i)
      f_val = result.x
      f_x = result.jacobian.data

      # HACK: folx only return hessian trace, whereas the current
      # API returns all terms in the hessian diagonal
      f_lapl = result.laplacian

      d = cfg.rand_batch_size or cfg.dim
      f_xx = f_lapl * jnp.ones(d) / d

      return idx_set, f_val, f_x, f_xx

    if cfg.hess_diag_method == "dense_stde":
      taylor_2 = lambda v: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((v, jnp.zeros(dim)),),
      )

      rand_sub_vec = get_hutchinson_random_vec(idx_set, cfg, with_time)

      f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_sub_vec)
      f_val = f_vals[0]

      if not with_time:
        trace_est = jnp.mean(hvps)
      else:
        trace_est = jnp.mean(hvps[:-1])

      # HACK: hte only return hessian trace, whereas the current
      # API returns all terms in the hessian diagonal
      d = cfg.rand_batch_size or cfg.dim

      if not with_time:
        f_xx = trace_est * jnp.ones(d) / d
      else:
        f_xx = trace_est * jnp.ones(d + 1) / d
        # add time derivative
        f_xx = f_xx.at[-1].set(hvps[-1])

      f_x = jax.grad(f_partial)(x_i)

      return idx_set, f_val, f_x, f_xx

    if with_time:
      assert cfg.rand_batch_size == 0
      # NOTE: make sure time dim is always sampled
      idx_set = jnp.concatenate(
        [idx_set, jnp.array([cfg.dim], dtype=jnp.int32)]
      )

    if cfg.hess_diag_method == "sparse_stde":
      taylor_2 = lambda i: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((jnp.eye(dim)[i], jnp.zeros(dim)),),
      )
      f_vals, (f_x, hess_diag_val) = jax.vmap(taylor_2)(idx_set)
      f_val = f_vals[0]

      if cfg.rand_batch_size and cfg.apply_sampling_correction:
        hess_diag_val *= dim / cfg.rand_batch_size
      if not cfg.rand_jac:
        f_x = jax.grad(f_partial)(x_i)
      return idx_set, f_val, f_x, hess_diag_val

    hess_diag_val: Float[types.NPArray, "xi_dim"]

    if cfg.hess_diag_method == "forward":
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x, f_hess_fn = jax.linearize(f_grad_fn, x_i)  # jvp over vjp
      f_hess_diag_fn = jax.checkpoint(lambda i: f_hess_fn(jnp.eye(dim)[i])[i])
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "stacked":
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x = f_grad_fn(x_i)
      f_hess_diag_fn = lambda i: hvp(f_partial, x_i, jnp.eye(dim)[i])[i]
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "scan":
      # NOTE: this is slower than vmap. Also idx_set need to be jnp.array
      # for tracing
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x = f_grad_fn(x_i)
      f_hess_diag_fn = lambda i: hvp(f_partial, x_i, jnp.eye(dim)[i])[i]
      _, hess_diag_val = lax.scan(
        lambda i, _: (i + 1, f_hess_diag_fn(idx_set[i])),
        0,
        None,
        length=len(idx_set),
      )

    else:
      raise ValueError

    if cfg.rand_batch_size and cfg.apply_sampling_correction:
      hess_diag_val *= dim / cfg.rand_batch_size  # sampling correction

    return idx_set, f_val, f_x, hess_diag_val

  return fn_hess_diag
