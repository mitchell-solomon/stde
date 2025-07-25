import jax
import jax.numpy as jnp
from stde.config import EqnConfig
import stde.equations as equations


def test_nls_shapes_and_residual():
    cfg = EqnConfig(dim=1)
    sample_fn = equations.NLS.get_sample_domain_fn(cfg)
    x, t, xb, tb, _ = sample_fn(3, 3, jax.random.PRNGKey(0))
    assert x.shape == (3, 1)
    assert t.shape == (3, 1)

    u_val = equations.NLS.sol(x, t, cfg)
    assert u_val.shape[0] == 3

    u_fn = lambda x_, t_: equations.NLS.sol(x_[None], t_[None], cfg)
    res = equations.NLS.res(x[0], t[0], u_fn, cfg, jax.random.PRNGKey(0))
    assert jnp.allclose(res, 0.0, atol=1e-5)


def test_nls_sample_domain_bounds():
    cfg = EqnConfig(dim=1)
    sample_fn = equations.NLS.get_sample_domain_fn(cfg)
    x, t, xb, tb, _ = sample_fn(100, 100, jax.random.PRNGKey(1))
    assert x.min() >= -5.0 and x.max() <= 5.0
    assert t.min() >= 0.0 and t.max() <= jnp.pi / 2
