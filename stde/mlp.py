from dataclasses import dataclass
from typing import Sequence

from stde.model import WaveAct

import jax.numpy as jnp
from jax import jit
import flax.linen as nn

_INIT_MAP = {
    "kaiming_uniform": nn.initializers.kaiming_uniform(),
    "xavier_normal": nn.initializers.xavier_normal(),
    "default": nn.initializers.lecun_normal(),
}

_BIAS_INIT_MAP = {
    "default": nn.initializers.zeros,
}


@dataclass(frozen=True)
class MlpConfig:
    """Configuration for ``MlpBackbone``."""
    width: int = 128
    depth: int = 4
    w_init: str = "kaiming_uniform"
    b_init: str = "default"
    block_size: int = -1
    use_conv: bool = False
    hidden_sizes: Sequence[int] = ()
    activation: str = "tanh"


@jit
def relu(x):
    return jnp.maximum(x, 0)

class MlpBackbone(nn.Module):
    cfg: MlpConfig
    time_dependent: bool = False

    def _activate(self, x):
        if self.cfg.activation == "gelu":
            return nn.gelu(x)
        elif self.cfg.activation == "relu":
            return relu(x)
        elif self.cfg.activation == "silu":
            return nn.silu(x)
        elif self.cfg.activation == "wave":
            return WaveAct()(x)
        elif self.cfg.activation == "tanh":
            return nn.tanh(x)
        else:
            raise ValueError(f"unknown activation {self.cfg.activation}")

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x: jax.Array
            Input of shape ``(B, L, D)``.
        """
        B, L, D = x.shape

        init_kwargs = {}
        if self.cfg.w_init in _INIT_MAP:
            init_kwargs["kernel_init"] = _INIT_MAP[self.cfg.w_init]
        else:
            init_kwargs["kernel_init"] = _INIT_MAP["default"]
        if self.cfg.b_init in _BIAS_INIT_MAP:
            init_kwargs["bias_init"] = _BIAS_INIT_MAP[self.cfg.b_init]
        else:
            init_kwargs["bias_init"] = _BIAS_INIT_MAP["default"]

        xt = x
        if self.cfg.block_size != -1:
            x_body = xt[..., :-1] if self.time_dependent else xt
            if self.cfg.use_conv:
                x_body = nn.Conv(
                    features=1,
                    kernel_size=(self.cfg.block_size,),
                    strides=(self.cfg.block_size,),
                    padding="VALID",
                    name="block_conv",
                    **init_kwargs,
                )(x_body[..., None])
                x_body = x_body[..., 0]
            else:
                B_, L_, D_ = x_body.shape
                x_body = x_body.reshape(B_ * L_, D_)
                if D_ % self.cfg.block_size != 0:
                    pad = self.cfg.block_size - (D_ % self.cfg.block_size)
                    x_body = jnp.pad(x_body, ((0, 0), (0, pad)))
                n_blocks = x_body.shape[1] // self.cfg.block_size
                x_body = x_body.reshape(B_ * L_, n_blocks, self.cfg.block_size)
                x_body = nn.Dense(1, name="linear_first", **init_kwargs)(x_body)
                x_body = self._activate(x_body)
                x_body = x_body[..., 0]
                x_body = x_body.reshape(B_, L_, n_blocks)
            if self.time_dependent:
                xt = jnp.concatenate([x_body, xt[..., -1:]], axis=-1)
            else:
                xt = x_body

        h = xt.reshape(B * L, xt.shape[-1])
        hidden_sizes = list(self.cfg.hidden_sizes) if self.cfg.hidden_sizes else [self.cfg.width] * (self.cfg.depth - 1)
        for i, size in enumerate(hidden_sizes):
            h = nn.Dense(size, name=f"dense_{i}", **init_kwargs)(h)
            h = self._activate(h)
        h = nn.Dense(1, name=f"dense_out", **init_kwargs)(h)
        out = h.reshape(B, L)
        return out
