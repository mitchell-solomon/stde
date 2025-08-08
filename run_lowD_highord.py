#!/usr/bin/env python
"""Run hyperparameter sweeps for low-dimensional high-order PDEs."""

import argparse
import itertools
import os
import subprocess
from pathlib import Path

import stde.equations as eqns

# Hyperparameter sweep spaces, mirroring run_backbone_sweep.py
MLP_SWEEP = {
    "mlp_depth": [1, 2],
    "mlp_width": [4, 64],
    "block_size": [32, 64],
    "activation": ["tanh", "gelu"],
    "ad_mode": ["reverse", "forward"],  # --ad_mode
    "no_stde": [False, True],            # --no_stde (store_true)
}

MAMBA_SWEEP = {
    "num_mamba_blocks": [1, 2],
    "hidden_features": [8, 16],
    "expansion_factor": [1.0],
    "dt_rank": ["auto"],
    "activation": ["tanh", "wave", "gelu"],
    "bidirectional": [True, False],
    "ad_mode": ["reverse", "forward"],  # --ad_mode
    "no_stde": [False, True],            # --no_stde (store_true)
}

# Variants of the highord1d equation controlled via env variables
HIGHORD1D_VARIANTS = {
    "sg": {"HIGHORD1D_CASE": "2", "HIGHORD1D_EQ": "1"},
    "gkdv": {"HIGHORD1D_CASE": "3", "HIGHORD1D_EQ": "2"},
    "gkdv_high": {"HIGHORD1D_CASE": "4", "HIGHORD1D_EQ": "3"},
}


def default_dim(name: str) -> int:
    if name == "KdV2d":
        return 2
    if name == "highord1d":
        return 1
    eqn = getattr(eqns, name)
    if name in {"SemilinearHeatTime", "SineGordonTime", "AllenCahnTime"}:
        return 10
    if "Threebody" in name:
        return 3
    return 1 if eqn.time_dependent else 2


def iter_sweep(backbone: str):
    grid = MLP_SWEEP if backbone == "MLP" else MAMBA_SWEEP
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        params = dict(zip(keys, values))
        if params.get("ad_mode") == "reverse" and not params.get("no_stde"):
            continue
        yield params


def main():
    parser = argparse.ArgumentParser(
        description="Run low-dimensional high-order PDE sweeps"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["highord1d"], #KdV2d
        help="equation names to run",
    )
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=50000000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--use_seed_seq", type=bool, default=True)
    parser.add_argument("--seed_frac", type=float, default=0.01)
    parser.add_argument(
        "--spatial_dim",
        type=int,
        default=None,
        help="override default spatial dimension for all equations",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("_results"),
        help="base directory to store results",
    )
    parser.add_argument("--overwrite", action="store_true", help="rerun if results exist")
    args, unknown = parser.parse_known_args()

    common_args = [
        "--epochs",
        str(args.epochs),
        "--eval_every",
        str(args.eval_every),
        "--lr",
        str(args.lr),
        "--N_test",
        str(args.n_test),
        "--test_batch_size",
        str(args.test_batch_size),
        "--seq_len",
        str(args.seq_len),
        "--use_seed_seq",
        str(args.use_seed_seq).lower(),
        "--seed_frac",
        str(args.seed_frac),
    ] + unknown

    BOOL_FLAGS = {
        "bidirectional", "complement", "tie_in_proj", "tie_gate",
        "diag_skip", "diag_gate", "diag_gated", "diag_residual",
        "recursive_scan", "recursive_split", "custom_vjp_scan",
        "no_stde",
    }

    for backbone in ["MLP"]:  # "Mamba" could be added if desired
        for params in iter_sweep(backbone):
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            for seed in range(args.seeds):
                for eqn_name in args.benchmarks:
                    variants = [(None, {})]
                    if eqn_name == "highord1d":
                        variants = HIGHORD1D_VARIANTS.items()
                    for variant_name, env_update in variants:
                        eqn_tag = eqn_name if variant_name is None else f"{eqn_name}_{variant_name}"
                        run_dir = args.results_dir / backbone / eqn_tag / param_str / str(seed)
                        final_path = run_dir / "final_eval_results.json"
                        if final_path.exists() and not args.overwrite:
                            print(f"Skipping {run_dir} (already exists)")
                            continue
                        print(f"Running {run_dir}")
                        run_name = f"{backbone}/{eqn_tag}/{param_str}/{seed}"
                        dim = args.spatial_dim if args.spatial_dim is not None else default_dim(eqn_name)
                        cmd = [
                            "python",
                            "-m",
                            "stde.train",
                            "--eqn_name",
                            eqn_name,
                            "--spatial_dim",
                            str(dim),
                            "--run_name",
                            run_name,
                            "--SEED",
                            str(seed),
                            "--backbone",
                            backbone,
                        ] + common_args
                        for k, v in params.items():
                            if k in BOOL_FLAGS:
                                if v:
                                    cmd.append(f"--{k}")
                            elif k == "backbone":
                                continue
                            else:
                                cmd.extend([f"--{k}", str(v)])
                        env = os.environ.copy()
                        env.update(env_update)
                        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
