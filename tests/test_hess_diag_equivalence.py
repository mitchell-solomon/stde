import jax
import jax.numpy as jnp

from stde.config import EqnConfig
from stde.operators import hess_diag


C = 3.0


def quadratic(x: jnp.ndarray) -> jnp.ndarray:
    """Simple quadratic with constant Hessian diagonal.

    The Hessian of ``0.5 * C * ||x||^2`` is ``C`` times the identity,
    so all estimation methods should agree on the diagonal values.
    """
    return 0.5 * C * jnp.sum(x ** 2)


def _run(method: str, key: jax.Array) -> jnp.ndarray:
    cfg = EqnConfig(
        dim=2,
        rand_batch_size=2,  # use full dimension to avoid subsampling effects
        hess_diag_method=method,
        stde_dist="rademacher",
        apply_sampling_correction=False,
    )
    hd_fn = hess_diag(quadratic, cfg)
    _, _, _, hess = hd_fn(jnp.array([1.0, -1.0]), key=key)
    return hess


def test_hess_diag_methods_equivalent():
    key = jax.random.PRNGKey(3)
    methods = ["sparse_stde", "dense_stde", "stacked", "forward", "scan"]
    results = {m: _run(m, key) for m in methods}
    expected = jnp.full_like(next(iter(results.values())), C)

    for diag in results.values():
        assert jnp.allclose(diag, expected)

    baseline = results[methods[0]]
    for m in methods[1:]:
        assert jnp.allclose(baseline, results[m])
