import inspect
import pytest
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
import haiku as hk
# import folx
from stde.config import EqnConfig
from stde.types import Equation
import stde.equations as equations

# Collect all Equation objects defined in equations.py
EQUATION_NAMES = [name for name, val in inspect.getmembers(equations)
                  if isinstance(val, Equation)]

@pytest.mark.parametrize('name', EQUATION_NAMES)
def test_equation_forms(name):
    eqn: Equation = getattr(equations, name)
    dim = 3 if 'Threebody' in name else 2
    cfg = EqnConfig(dim=dim)

    if getattr(eqn, 'random_coeff', False):
        cfg.coeffs = jnp.ones((1, cfg.dim))

    sample_fn = eqn.get_sample_domain_fn(cfg)
    x, t, xb, tb, _ = sample_fn(2, 2, jax.random.PRNGKey(0))

    # Shapes of sampled points
    assert x.shape[0] == 2
    if t is not None:
        assert t.shape[0] == 2

    # Solution and boundary condition should produce outputs matching batch size
    u_val = eqn.sol(x, t, cfg)
    assert u_val.shape[0] == x.shape[0]

    g_val = eqn.boundary_cond(xb, tb, cfg)
    assert g_val.shape[0] == xb.shape[0]

    enforced = eqn.enforce_boundary(xb, tb, g_val, cfg)
    assert enforced.shape == g_val.shape

    # Residual should vanish for the analytical solution if defined on the domain
    if not getattr(eqn, 'is_traj', False):
        u_fn = lambda x_, t_: eqn.sol(x_[None], t_ if t_ is None else t_[None], cfg)
        res = eqn.res(x[0], None if t is None else t[0], u_fn, cfg)
        assert jnp.allclose(res, 0.0, atol=1e-4)


if __name__ == "__main__":
    for name in EQUATION_NAMES:
        test_equation_forms(name)
        print(f"Test passed for {name}")
    
    print("All tests passed!")
