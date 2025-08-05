#!/usr/bin/env python
"""Run hyperparameter sweeps for MLP and Mamba backbones."""


import argparse
import itertools
import subprocess
from pathlib import Path
import stde.equations as eqns


# Hyperparameter sweep spaces, now including ad_mode and no_stde
MLP_SWEEP = {
    "mlp_depth": [1, 2],
    "mlp_width": [4, 64],
    "block_size": [32, 64],
    "activation": ["tanh", "gelu"],
    "ad_mode": ["reverse", "forward"],  # --ad_mode
    "no_stde": [False, True],           # --no_stde (store_true)
}

MAMBA_SWEEP = {
    "num_mamba_blocks": [1, 2],
    "hidden_features": [8, 16],
    "expansion_factor": [2.0],
    "dt_rank": ["auto"],
    "activation": ["tanh", "gelu", "wave", "relu"],
    "bidirectional": [True, False],
    "ad_mode": ["reverse", "forward"],  # --ad_mode
    "no_stde": [False, True],           # --no_stde (store_true)
}


def default_dim(name: str) -> int:
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
        yield params


def main():
    parser = argparse.ArgumentParser(description="Run backbone hyperparameter sweeps or ablation variants")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            # "Poisson",
            "SineGordonTwobody",
            # "Wave",
            # "Burgers",
            # "KdV2d",
            # "PoissonHouman",
            # "SineGordonTime",
            # "AllenCahnTime",
            # "SemilinearHeatTime",
        ],
        help="equation names to run",
    )
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=50000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--use_seed_seq", type=bool, default=True)
    parser.add_argument("--seed_frac", type=float, default=0.01)
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

    # List of boolean flags in train.py that use action="store_true"
    BOOL_FLAGS = {
        "bidirectional", "complement", "tie_in_proj", "tie_gate",
        "diag_skip", "diag_gate", "diag_gated", "diag_residual",
        "recursive_scan", "recursive_split", "custom_vjp_scan",
        "no_stde"
    }

    # Run hyperparameter sweep (including ablation variants as part of the grid)
    for backbone in ["Mamba"]: #"MLP", 
        for params in iter_sweep(backbone):
            # Compose a unique string for the run directory
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            for seed in range(args.seeds):
                for eqn_name in args.benchmarks:
                    run_dir = args.results_dir / backbone / eqn_name / param_str / str(seed)
                    final_path = run_dir / "final_eval_results.json"
                    if final_path.exists() and not args.overwrite:
                        print(f"Skipping {run_dir} (already exists)")
                        continue
                    print(f"Running {run_dir}")
                    run_name = f"{backbone}/{param_str}/{eqn_name}/{seed}"
                    dim = default_dim(eqn_name)
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
                    # Add sweep params, skipping backbone (already present)
                    for k, v in params.items():
                        if k in BOOL_FLAGS:
                            if v:
                                cmd.append(f"--{k}")
                        elif k == "backbone":
                            continue  # already included
                        else:
                            cmd.extend([f"--{k}", str(v)])
                    # print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
