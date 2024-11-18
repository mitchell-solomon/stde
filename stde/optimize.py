from typing import Dict, Tuple

import haiku as hk
import jax
import optax

from .config import GDConfig
from .types import TrainingState


def get_optimizer(
  cfg: GDConfig,
  params: hk.Params,
  rng_key: jax.Array,
) -> Dict[str, Tuple[optax.GradientTransformation, TrainingState]]:
  opt_states = dict()
  if cfg.lr_decay == "piecewise":
    lr = optax.piecewise_constant_schedule(
      init_value=cfg.lr,
      boundaries_and_scales={
        int(cfg.epochs * 0.5): 0.5,
        int(cfg.epochs * 0.75): 0.25,
        int(cfg.epochs * 0.825): 0.125,
      }
    )
  elif cfg.lr_decay == "linear":
    lr = optax.linear_schedule(
      init_value=cfg.lr, end_value=0., transition_steps=cfg.epochs
    )
  elif cfg.lr_decay == "exponential":
    lr = optax.exponential_decay(
      init_value=cfg.lr, transition_steps=cfg.epochs, decay_rate=cfg.gamma
    )
  elif cfg.lr_decay == "cosine":
    lr = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=1.0,
      warmup_steps=50,
      decay_steps=cfg.epochs - 50,
      end_value=0.0,
    )
  else:
    lr = cfg.lr
  optimizer = getattr(optax, cfg.optimizer)(learning_rate=lr)

  opt_state = optimizer.init(params)
  state = TrainingState(params, opt_state, rng_key)

  opt_states["main"] = (optimizer, state)

  return opt_states
