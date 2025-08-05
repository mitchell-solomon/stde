"""Generate test and validation datasets for all equations.

This script replicates the dataset generation logic found in
``stde/train.py`` lines 660-704. For each equation defined in
:class:`stde.config.EqnConfig`, it creates validation and test datasets for
five random seeds and saves them under ``data/{equation}/{seed}/``.

Each directory contains four files stored with ``jnp.save``:
``test_seqs.npy``, ``test_truths.npy``, ``val_seqs.npy`` and
``val_truths.npy``. Only equations that are not trajectory-based are
processed.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from stde.config import EqnConfig
import stde.equations as eqns


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def default_dim(name: str) -> int:
    """Return a reasonable default spatial dimension for an equation."""
    eqn = getattr(eqns, name)
    if name in {"SemilinearHeatTime", "SineGordonTime", "AllenCahnTime"}:
        return 10
    if "Threebody" in name:
        return 3
    return 1 if eqn.time_dependent else 2


# ---------------------------------------------------------------------------
# Core dataset generation logic
# ---------------------------------------------------------------------------

def generate_for_equation(eqn_name: str, seeds: int = 5, data_dir: Path = Path("data")) -> None:
    eqn = getattr(eqns, eqn_name)
    if eqn.is_traj:
        print(f"Skipping {eqn_name} (trajectory-based equation)")
        return

    dim = default_dim(eqn_name)
    spatial_dim = dim
    N_test, test_batch_size = 2000, 20
    N_val, val_batch_size = 200, 20
    seq_len = 3
    use_seed_seq = True
    seed_frac = 0.01

    n_test_batches = N_test // test_batch_size
    n_val_batches = N_val // val_batch_size

    for seed in range(seeds):
        eqn_cfg = EqnConfig(name=eqn_name, dim=dim)
        if eqn.random_coeff:
            coeff_rng = np.random.default_rng(seed)
            eqn_cfg.coeffs = coeff_rng.standard_normal((1, dim))
        sample_domain_fn = eqn.get_sample_domain_fn(eqn_cfg)

        if eqn.time_dependent:
            def sol_fn(xt):
                x_part = xt[..., :spatial_dim]
                t_part = xt[..., spatial_dim:]
                return eqn.sol(x_part, t_part, eqn_cfg)
        else:
            def sol_fn(x):
                return eqn.sol(x, None, eqn_cfg)

        # compute domain span for seed-based sequence sampling
        span_rng = jax.random.PRNGKey(seed + 1)
        x_tmp, t_tmp, _xb, _tb, _ = sample_domain_fn(1024, 8, span_rng)
        x_span = float(jnp.max(x_tmp) - jnp.min(x_tmp))
        if eqn.time_dependent and t_tmp is not None:
            t_span = float(jnp.max(t_tmp) - jnp.min(t_tmp))
        else:
            t_span = 0.0
        seed_x_sigma = seed_frac * x_span
        seed_t_sigma = seed_frac * t_span

        @partial(jax.jit, static_argnames=("batch_size", "seq_len", "use_seed"))
        def sample_domain_seq_fn(batch_size: int, rng: jax.Array, seq_len: int, use_seed: bool):
            if use_seed and not eqn.is_traj:
                x_seed, t_seed, _xb, _tb, rng = sample_domain_fn(batch_size, 0, rng)
                keys = jax.random.split(
                    rng, 3 if eqn.time_dependent and t_seed is not None else 2
                )
                rng = keys[-1]
                x_noise = seed_x_sigma * jax.random.normal(
                    keys[0], (batch_size, seq_len, spatial_dim)
                )
                x_seq = x_seed[:, None, :] + x_noise
                if eqn.time_dependent and t_seed is not None:
                    t_noise = seed_t_sigma * jax.random.normal(
                        keys[1], (batch_size, seq_len, 1)
                    )
                    t_seq = t_seed[:, None, :] + t_noise
                    sort_idx = jnp.argsort(t_seq[..., 0], axis=1)
                    x_seq = jnp.take_along_axis(x_seq, sort_idx[..., None], axis=1)
                    t_seq = jnp.take_along_axis(t_seq, sort_idx[..., None], axis=1)
                    x_seq = jnp.concatenate([x_seq, t_seq], axis=-1)
                return x_seq, rng
            else:
                if eqn.is_traj:
                    x, t, _xb, _tb, rng = sample_domain_fn(batch_size, seq_len - 1, rng)
                else:
                    x, t, _xb, _tb, rng = sample_domain_fn(batch_size * seq_len, 0, rng)
                x_seq = x.reshape((batch_size, seq_len, -1))
                if eqn.time_dependent and t is not None:
                    t_seq = t.reshape((batch_size, seq_len, -1))
                    sort_idx = jnp.argsort(t_seq[..., 0], axis=1)
                    x_seq = jnp.take_along_axis(x_seq, sort_idx[..., None], axis=1)
                    t_seq = jnp.take_along_axis(t_seq, sort_idx[..., None], axis=1)
                    x_seq = jnp.concatenate([x_seq, t_seq], axis=-1)
                return x_seq, rng

        rng = jax.random.PRNGKey(seed)

        # Test set generation
        test_seqs, test_truths = [], []
        for _ in tqdm(range(n_test_batches), desc=f"{eqn_name} seed {seed} test"):
            rng, sample_rng = jax.random.split(rng)
            x_test_seq, rng = sample_domain_seq_fn(
                batch_size=test_batch_size,
                rng=sample_rng,
                seq_len=seq_len,
                use_seed=use_seed_seq,
            )
            B, L, D = x_test_seq.shape
            x_flat = x_test_seq.reshape((B * L, D))
            y_flat = jax.vmap(sol_fn)(x_flat)
            test_seqs.append(x_test_seq)
            test_truths.append(y_flat.reshape((B, L)))
        test_seqs = jnp.stack(test_seqs)
        test_truths = jnp.stack(test_truths)

        # Validation set generation
        val_seqs, val_truths = [], []
        for _ in tqdm(range(n_val_batches), desc=f"{eqn_name} seed {seed} val"):
            rng, sample_rng = jax.random.split(rng)
            x_val_seq, rng = sample_domain_seq_fn(
                batch_size=val_batch_size,
                rng=sample_rng,
                seq_len=seq_len,
                use_seed=use_seed_seq,
            )
            B, L, D = x_val_seq.shape
            x_flat = x_val_seq.reshape((B * L, D))
            y_flat = jax.vmap(sol_fn)(x_flat)
            val_seqs.append(x_val_seq)
            val_truths.append(y_flat.reshape((B, L)))
        val_seqs = jnp.stack(val_seqs)
        val_truths = jnp.stack(val_truths)

        # Save arrays
        out_dir = data_dir / eqn_name / str(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        jnp.save(out_dir / "test_seqs.npy", test_seqs)
        jnp.save(out_dir / "test_truths.npy", test_truths)
        jnp.save(out_dir / "val_seqs.npy", val_seqs)
        jnp.save(out_dir / "val_truths.npy", val_truths)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    eqn_names = get_args(EqnConfig.__annotations__["name"])
    for eqn_name in eqn_names:
        generate_for_equation(eqn_name)


if __name__ == "__main__":
    main()
