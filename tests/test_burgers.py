import jax
import jax.numpy as jnp

from stde.config import EqnConfig
from stde import equations


def test_burgers_zero_residual():
    cfg = EqnConfig(dim=1)
    eqn = equations.Burgers

    def u_fn(x, t):
        return eqn.sol(x, t, cfg)

    # sample a grid of points
    x = jnp.linspace(-1, 1, 5).reshape(-1, 1)
    t = jnp.linspace(0, cfg.T, 5).reshape(-1, 1)

    keys = jax.random.split(jax.random.PRNGKey(0), x.shape[0])
    res = jax.vmap(lambda x_, t_, k: eqn.res(x_, t_, u_fn, cfg, k))(x, t, keys)
    assert jnp.allclose(res, 0.0, atol=1e-5)
