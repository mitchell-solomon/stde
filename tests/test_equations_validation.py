import jax
import jax.numpy as jnp
import pytest

from stde.config import EqnConfig
import stde.equations as equations

# PDEs with closed-form solutions and boundary conditions
PDE_NAMES = ["HJB_LIN", "HJB_LQG", "BSB", "Wave", "Poisson"]
RES_PDE_NAMES = ["HJB_LIN", "BSB", "Wave", "Poisson"]


BOUNDARY_TIME = {
    "HJB_LIN": lambda cfg: cfg.T,
    "HJB_LQG": lambda cfg: cfg.T,
    "BSB": lambda cfg: cfg.T,
    "Wave": lambda cfg: 0.0,
    "Poisson": lambda cfg: 0.0,
}



@pytest.mark.parametrize("name", PDE_NAMES)
def test_boundary_condition_matches_solution(name):
    cfg = EqnConfig(name=name, dim=2, mc_batch_size=100, hess_diag_method="forward")
    eqn = getattr(equations, name)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, cfg.dim))
    t_val = BOUNDARY_TIME[name](cfg)
    t = jnp.ones((5, 1)) * t_val

    bc = eqn.boundary_cond(x, t, cfg)
    sol = eqn.sol(x, t, cfg)
    assert jnp.allclose(bc, sol, atol=1e-5)


@pytest.mark.parametrize("name", RES_PDE_NAMES)
def test_exact_solution_zero_residual(name):
    cfg = EqnConfig(name=name, dim=2, mc_batch_size=100, hess_diag_method="forward")
    eqn = getattr(equations, name)

    sample_fn = eqn.get_sample_domain_fn(cfg)
    x, t, _xb, _tb, key = sample_fn(4, 4, jax.random.PRNGKey(1))

    u_fn = lambda x_, t_: jnp.squeeze(eqn.sol(x_[None], t_[None], cfg))
    keys = jax.random.split(key, x.shape[0])
    res = jax.vmap(lambda x_, t_, k: eqn.res(x_, t_, u_fn, cfg, k))(x, t, keys)
    assert jnp.allclose(res, 0.0, atol=1e-5)


@pytest.mark.parametrize("name", RES_PDE_NAMES)
def test_end_to_end_loss_zero_for_exact_solution(name):
    cfg = EqnConfig(name=name, dim=2, mc_batch_size=100, hess_diag_method="forward")
    eqn = getattr(equations, name)

    sample_fn = eqn.get_sample_domain_fn(cfg)
    x, t, xb, tb, key = sample_fn(5, 5, jax.random.PRNGKey(2))

    u_fn = lambda x_, t_: jnp.squeeze(eqn.sol(x_[None], t_[None], cfg))
    keys = jax.random.split(key, x.shape[0])
    residuals = jax.vmap(lambda x_, t_, k: eqn.res(x_, t_, u_fn, cfg, k))(x, t, keys)
    res_loss = jnp.mean(residuals**2)

    bc_true = eqn.boundary_cond(xb, tb, cfg)
    bc_pred = eqn.sol(xb, tb, cfg)
    bc_loss = jnp.mean((bc_true - bc_pred) ** 2)

    assert res_loss < 1e-6
    assert bc_loss < 1e-6
