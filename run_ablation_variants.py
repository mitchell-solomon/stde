#!/usr/bin/env python
"""Run defined ablation variants across benchmarks and random seeds."""
import argparse
import subprocess
from pathlib import Path
import stde.equations as eqns


# Mapping of variant names to the stde.train CLI arguments
VARIANTS = {
    # Standard PINN baseline: MLP backbone, reverse-mode AD, no STDE
    "A1": ["--backbone", "MLP", "--no_stde", "--ad_mode", "reverse"],
    # SSM backbone only: Bi-Mamba backbone, reverse-mode AD, no STDE
    "A2": ["--backbone", "Mamba", "--no_stde", "--ad_mode", "reverse"],
    # Forward-mode AD without stochasticity
    "A3": ["--backbone", "MLP", "--no_stde", "--ad_mode", "forward"],
    # STDE estimator only (forward AD used internally)
    "A4": ["--backbone", "MLP"],
    # Backbone + forward AD
    "A5": ["--backbone", "Mamba", "--no_stde", "--ad_mode", "forward"],
    # Full model: Mamba backbone + forward AD + STDE
    "A6": ["--backbone", "Mamba"],
}


def default_dim(name: str) -> int:
    """Return a reasonable spatial dimension for a PDE."""
    eqn = getattr(eqns, name)
    if name in {"SemilinearHeatTime", "SineGordonTime", "AllenCahnTime"}:
        return 10
    if "Threebody" in name:
        return 3
    return 1 if eqn.time_dependent else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation variants")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["Poisson", "Wave"], # ["Burgers", "Poisson", "Wave", "KdV2d"]
        help="equation names to run",
    )
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=10000, help="training epochs")
    parser.add_argument("--eval_every", type=int, default=50000000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_test", type=int, default=20000)
    parser.add_argument("--test_batch_size", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--use_seed_seq", type=bool, default=True)
    parser.add_argument("--seed_frac", type=float, default=0.01)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("_results"),
        help="base directory to store results",
    )

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

    for variant, var_args in VARIANTS.items():
        for seed in range(args.seeds):
            for eqn_name in args.benchmarks:
                run_name = f"{variant}/{eqn_name}/{seed}"
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
                ] + common_args + var_args
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
