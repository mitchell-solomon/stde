import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from stde.config import EqnConfig
from stde.operators import get_hutchinson_random_vec, get_sdgd_idx_set


def test_get_hutchinson_random_vec_shapes(monkeypatch):
    cfg = EqnConfig(dim=5, rand_batch_size=3)
    key = jax.random.PRNGKey(0)

    key_idx, key = jax.random.split(key)
    idx_set = get_sdgd_idx_set(cfg, key=key_idx)
    key_vec, key = jax.random.split(key)
    vec = get_hutchinson_random_vec(idx_set, cfg, key=key_vec)
    assert vec.shape == (cfg.rand_batch_size, cfg.dim)

    key_time, _ = jax.random.split(key)
    vec_time = get_hutchinson_random_vec(idx_set, cfg, key=key_time, with_time=True)
    assert vec_time.shape == (cfg.rand_batch_size + 1, cfg.dim + 1)
    assert jnp.all(vec_time[-1] == jnp.eye(cfg.dim + 1)[-1])


if __name__ == "__main__":
    test_get_hutchinson_random_vec_shapes()