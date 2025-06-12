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
  """Return a function that estimates the Hessian trace of ``fn``.

  The returned function evaluates ``fn`` and computes a Hutchinson
  estimator of the trace of the Hessian with respect to the ``argnums``-th
  argument.  Only a subset of coordinates is used when ``cfg.rand_batch_size``
  is non‑zero.  The estimation relies on JAX's ``jet`` API to obtain
  Hessian‑vector products.

  Returns a callable ``f_trace(*xs)`` that outputs

  ``idx_set`` : the sampled indices used for the estimator.
  ``f_val``   : the scalar function value.
  ``trace_est`` : the estimated Hessian trace.
  """

  def fn_trace(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "1"],
  ]:
    # Extract the argument we differentiate with respect to and
    # record its dimension for creating zero-series terms below.
    x_i = xs[argnums]
    dim = cfg.dim

    # Fix all other arguments so that ``f_partial`` depends only on ``x_i``.
    f_partial = partial_i(fn, argnums, *xs)

    # Randomly select a subset of coordinates for the Hutchinson estimator.
    idx_set = get_sdgd_idx_set(cfg)

    # Draw random vectors supported on ``idx_set``.
    rand_sub_vec = get_hutchinson_random_vec(idx_set, cfg)

    # Build a second-order Taylor expansion using ``jet`` to obtain
    # Hessian-vector products for each random vector ``v``.
    taylor_2 = lambda v: jet.jet(
      fun=f_partial,
      primals=(x_i,),
      series=((v, jnp.zeros(dim)),),
    )

    # Vectorize over the random vectors to compute all Hv products at once.
    f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_sub_vec)

    # Average the Hutchinson estimates to approximate the Hessian trace.
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
    # extract the argument we want to differentiate with respect to
    x_i = xs[argnums]
    # append an additional dimension if time is included in x_i
    dim = cfg.dim if not with_time else cfg.dim + 1

    # create a partial function in which only x_i is a variable
    f_partial = partial_i(fn, argnums, *xs)
    # indices of x_i along which the hessian diagonal is computed
    idx_set = get_sdgd_idx_set(cfg)

    if cfg.hess_diag_method == "folx":
      # the folx package provides a forward-mode Laplacian operator
      # that yields function value, gradient and trace of the Hessian
      assert not with_time

      fwd_f = forward_laplacian(f_partial)
      result = fwd_f(x_i)
      f_val = result.x
      f_x = result.jacobian.data

      # HACK: folx only returns the Hessian trace.  The API of this
      # function expects the full diagonal, so broadcast the trace
      # value across all dimensions.
      f_lapl = result.laplacian

      d = cfg.rand_batch_size or cfg.dim
      f_xx = f_lapl * jnp.ones(d) / d

      return idx_set, f_val, f_x, f_xx

    if cfg.hess_diag_method == "dense_stde":
      # compute Hessian-vector products using the jet API over
      # randomly generated Hutchinson vectors
      taylor_2 = lambda v: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((v, jnp.zeros(dim)),),
      )

      # generate random vectors on the sampled subset of dimensions
      rand_sub_vec = get_hutchinson_random_vec(idx_set, cfg, with_time)

      # evaluate the function and its Hessian-vector products
      f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_sub_vec)
      f_val = f_vals[0]

      # average the Hutchinson estimators to obtain the trace estimate
      if not with_time:
        trace_est = jnp.mean(hvps)
      else:
        trace_est = jnp.mean(hvps[:-1])

      # HACK: hte only returns the Hessian trace. Here we broadcast the
      # trace estimate to form a diagonal vector.
      d = cfg.rand_batch_size or cfg.dim

      if not with_time:
        f_xx = trace_est * jnp.ones(d) / d
      else:
        f_xx = trace_est * jnp.ones(d + 1) / d
        # add time derivative
        f_xx = f_xx.at[-1].set(hvps[-1])  # keep exact time second derivative

      # compute gradient with respect to x_i
      f_x = jax.grad(f_partial)(x_i)

      return idx_set, f_val, f_x, f_xx

    if with_time:
      assert cfg.rand_batch_size == 0
      # ensure the time dimension is always included in the sample
      idx_set = jnp.concatenate(
        [idx_set, jnp.array([cfg.dim], dtype=jnp.int32)]
      )

    if cfg.hess_diag_method == "sparse_stde":
      # compute the Hessian diagonal explicitly one dimension at a time
      taylor_2 = lambda i: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((jnp.eye(dim)[i], jnp.zeros(dim)),),
      )
      f_vals, (f_x, hess_diag_val) = jax.vmap(taylor_2)(idx_set)
      f_val = f_vals[0]

      if cfg.rand_batch_size and cfg.apply_sampling_correction:
        # correct for sub-sampling of the Hessian entries
        hess_diag_val *= dim / cfg.rand_batch_size
      if not cfg.rand_jac:
        # evaluate the full gradient if it was not estimated jointly
        f_x = jax.grad(f_partial)(x_i)
      return idx_set, f_val, f_x, hess_diag_val

    hess_diag_val: Float[types.NPArray, "xi_dim"]

    if cfg.hess_diag_method == "forward":
      # compute the Hessian diagonal using a forward-over-reverse
      # approach which performs a jvp over a vjp
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x, f_hess_fn = jax.linearize(f_grad_fn, x_i)  # jvp over vjp
      f_hess_diag_fn = jax.checkpoint(lambda i: f_hess_fn(jnp.eye(dim)[i])[i])
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "stacked":
      # compute the Hessian diagonal by stacking reverse-mode Hessian
      # vector products for each coordinate
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x = f_grad_fn(x_i)
      f_hess_diag_fn = lambda i: hvp(f_partial, x_i, jnp.eye(dim)[i])[i]
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "scan":
      # sequentially scan over dimensions. This is slower than vmap
      # but avoids materializing the full Jacobian for large dim.
      # idx_set must be a JAX array for tracing
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
      # unknown method specified
      raise ValueError

    if cfg.rand_batch_size and cfg.apply_sampling_correction:
      # account for the fact that only a subset of dimensions was used
      hess_diag_val *= dim / cfg.rand_batch_size  # sampling correction

    # return the sampled indices along with function value, gradient and
    # estimated Hessian diagonal
    return idx_set, f_val, f_x, hess_diag_val

  return fn_hess_diag
