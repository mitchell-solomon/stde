#!/usr/bin/env python
"""Run hyperparameter sweeps for MLP and Mamba backbones."""

import argparse
import itertools
import subprocess
from pathlib import Path
import stde.equations as eqns


MLP_SWEEP = {
    "mlp_depth": [2, 4],
    "mlp_width": [64, 128],
    "block_size": [32, 64],
    "activation": ["tanh", "relu"],
}

MAMBA_SWEEP = {
    "num_mamba_blocks": [1, 2],
    "hidden_features": [64, 128],
    "expansion_factor": [2.0, 4.0],
    "dt_rank": ["auto", "full"],
    "activation": ["tanh", "silu"],
    "bidirectional": [True, False],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backbone hyperparameter sweeps")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "Poisson",
            "Wave",
            "Burgers",
            "KdV2d",
            "PoissonHouman",
            "SineGordonTime",
            "AllenCahnTime",
            "SemilinearHeatTime",
        ],
        help="equation names to run",
    )
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=50000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_test", type=int, default=20000)
    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=5)
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

    for backbone in ["MLP", "Mamba"]:
        for params in iter_sweep(backbone):
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            for seed in range(args.seeds):
                for eqn_name in args.benchmarks:
                    run_dir = args.results_dir / backbone / param_str / eqn_name / str(seed)
                    final_path = run_dir / "final_eval_results.json"
                    if final_path.exists() and not args.overwrite:
                        print(f"Skipping {run_dir} (already exists)")
                        continue

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
                    for k, v in params.items():
                        if isinstance(v, bool):
                            if v:
                                cmd.append(f"--{k}")
                        else:
                            cmd.extend([f"--{k}", str(v)])
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
