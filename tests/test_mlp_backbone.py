import jax
import jax.numpy as jnp
import flax.linen as nn

from stde.mlp import MlpBackbone, MlpConfig


class InlineMLP(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape
        h = x.reshape(B * L, D)
        for _ in range(self.depth - 1):
            h = nn.Dense(self.width)(h)
            h = nn.tanh(h)
        h = nn.Dense(1)(h)
        return h.reshape(B, L)


def test_mlp_backbone_shape_matches_inline():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 3, 4))

    cfg = MlpConfig(width=8, depth=3)
    mlp = MlpBackbone(cfg)
    params = mlp.init(key, x)
    out = mlp.apply(params, x)

    inline = InlineMLP(width=8, depth=3)
    p2 = inline.init(key, x)
    out2 = inline.apply(p2, x)

    assert out.shape == out2.shape
