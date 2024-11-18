import dataclasses
import math
from collections import namedtuple
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random

import stde.types as types
from stde.config import EqnConfig, ModelConfig
from stde.equations import pinn_loss_fn
from stde.types import Equation


def calcualte_gain_leaky_relu(negative_slope: float) -> float:
  """This is what pytorch uses by default for the gain in
  kaiming uniform initialization for the Linear layer.

  Effective this samples from
  uniform(-1/sqrt(in_features), 1/sqrt(in_features))
  """
  return math.sqrt(2.0 / (1 + negative_slope**2))


INITIALIZERS = dict(
  kaiming_uniform=hk.initializers.VarianceScaling(
    scale=calcualte_gain_leaky_relu(math.sqrt(5))**2,
    mode='fan_in',
    distribution='uniform'
  ),
  xavier_normal=hk.initializers.VarianceScaling(
    scale=1., mode='fan_avg', distribution='normal'
  )
)


@dataclasses.dataclass
class PINN(hk.Module):
  eqn: Equation
  eqn_cfg: EqnConfig
  model_cfg: ModelConfig

  def __call__(self, x: types.X, t: types.T) -> jax.Array:
    """ansatze for space-time domain scalar function"""
    # squeeze to ensure scalar output
    inputs = x if not self.eqn.time_dependent else jnp.concatenate(
      [x, t], axis=-1
    )
    pred = jnp.squeeze(self.net(inputs))
    if self.eqn_cfg.enforce_boundary:
      return self.eqn.enforce_boundary(x, t, pred, self.eqn_cfg)
    else:
      return pred

  def net(self, xt: jax.Array) -> jax.Array:
    """NOTE: if eqn.time_dependent, the last dim is t."""
    if len(self.model_cfg.hidden_sizes) == 0:
      hidden_sizes = [self.model_cfg.width] * (self.model_cfg.depth - 1) + [1]
    else:
      hidden_sizes = list(self.model_cfg.hidden_sizes) + [1]

    init_kwargs = dict()
    if self.model_cfg.w_init in INITIALIZERS:
      init_kwargs['w_init'] = INITIALIZERS[self.model_cfg.w_init]
    if self.model_cfg.b_init in INITIALIZERS:
      init_kwargs['b_init'] = INITIALIZERS[self.model_cfg.b_init]

    if self.model_cfg.block_size != -1:  # WEIGHT SHARING VIA CONV
      if self.eqn.time_dependent:
        x = xt[..., :-1]
      else:
        x = xt

      if self.model_cfg.use_conv:
        x = hk.Conv1D(
          output_channels=1,
          kernel_shape=self.model_cfg.block_size,
          stride=self.model_cfg.block_size,
          padding="VALID"
        )(
          x[..., None]  # add channel dim
        )
        x = x[..., 0]  # remove channel dim

      else:  # use MLP
        x = x.reshape(xt.shape[:-1] + (-1, self.model_cfg.block_size))
        layer_fn = hk.Linear(1, name=f"linear_first", **init_kwargs)
        x = jax.nn.tanh(layer_fn(x))
        x = x[..., 0]  # remove channel dim

      if self.eqn.time_dependent:
        xt = jnp.concatenate([x, xt[..., -1:]], axis=-1)
      else:
        xt = x

    for i, h_i in enumerate(hidden_sizes):
      layer_fn = hk.Linear(h_i, name=f"linear_{i}", **init_kwargs)
      xt = layer_fn(xt)
      if i != len(hidden_sizes) - 1:  # activation
        xt = jax.nn.tanh(xt)
    return xt

  def loss_fn(self, x, t, x_boundary, t_boundary):
    return pinn_loss_fn(
      x, t, x_boundary, t_boundary, self.__call__, self.eqn_cfg, self.eqn
    )

  def err_norms_fn(self, x, t, y, y_t):
    y_pred = self.__call__(x, t)
    err = y - y_pred
    l1 = jnp.abs(err).sum()
    l2 = (err**2).sum()
    w1_t = 0.
    if self.eqn.time_dependent and self.model_cfg.compute_w1_loss:
      u_t = jax.vmap(jax.grad(self.__call__, argnums=1))(x, t)
      w1_t = ((u_t - y_t)**2).sum()
    return l1, l2, w1_t

  def init_for_multitransform(self):
    return (
      self.__call__,
      namedtuple("PINN", ["u", "loss_fn", "err_norms_fn"
                         ])(self.__call__, self.loss_fn, self.err_norms_fn),
    )
