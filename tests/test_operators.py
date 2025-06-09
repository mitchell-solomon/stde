import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
hk = pytest.importorskip('haiku')
pytest.importorskip('folx')

from stde.config import EqnConfig
from stde.operators import get_hutchinson_random_vec, get_sdgd_idx_set


def test_get_hutchinson_random_vec_shapes(monkeypatch):
    cfg = EqnConfig(dim=5, rand_batch_size=3)
    rng_seq = hk.PRNGSequence(0)
    monkeypatch.setattr(hk, 'next_rng_key', lambda: next(rng_seq))

    idx_set = get_sdgd_idx_set(cfg)
    vec = get_hutchinson_random_vec(idx_set, cfg)
    assert vec.shape == (cfg.rand_batch_size, cfg.dim)

    vec_time = get_hutchinson_random_vec(idx_set, cfg, with_time=True)
    assert vec_time.shape == (cfg.rand_batch_size + 1, cfg.dim + 1)
    assert jnp.all(vec_time[-1] == jnp.eye(cfg.dim + 1)[-1])
