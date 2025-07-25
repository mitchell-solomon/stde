"""Utilities to aggregate experiment outputs and analyze ablations."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def aggregate_results(results_dir: str = "_results") -> pd.DataFrame:
    """Recursively load all runs under ``results_dir`` into a DataFrame."""
    base = Path(results_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"results directory '{results_dir}' not found")

    records = []
    for cfg_path in base.rglob("config.json"):
        run_dir = cfg_path.parent
        res_path = run_dir / "final_eval_results.json"
        if not res_path.exists():
            continue

        with cfg_path.open() as f:
            config = json.load(f)
        with res_path.open() as f:
            results = json.load(f)

        rel_run = str(run_dir.relative_to(base))
        record = {"run": rel_run}
        record.update(config)
        record.update(results)
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["variant"] = df["run"].apply(lambda r: Path(r).parts[0])
        df["seed"] = df.get("SEED", df["run"].apply(lambda r: Path(r).parts[1] if len(Path(r).parts) > 1 else -1))
    return df


# seaborn box a whisker plots, grouped by equations then variant
def plot_boxplot(df, metric_col, out_file="aggregate_bw_plot.png"):
    """Create a box plot of ``metric_col`` grouped by eqn_name and variant."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="eqn_name", y=metric_col, hue="variant")
    plt.yscale('log')
    plt.xlabel("Equation")
    plt.ylabel(metric_col)
    plt.title("Evaluation Results")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


def plot_scatter(df, xcol, ycol, out_file="aggregate_plot.png"):
    """Create an annotated scatter plot of metric1 vs metric2, colored by variant."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=xcol, y=ycol, hue="variant")
    plt.xscale('log')
    plt.yscale('log')

    # Annotate each point with the run name
    for _, row in df.iterrows():
        label = row.get("eqn_name", row["run"])
        plt.text(row[xcol], row[ycol], label, fontsize=8,
                 ha="left", va="bottom")

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title("Evaluation Results")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


def summarize_variants(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Return mean and std of ``metrics`` grouped by variant."""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("variant")[list(metrics)].agg(["mean", "std"])
    # flatten column index
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def paired_ttests(
    df: pd.DataFrame,
    baseline: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Perform paired t-tests comparing each variant to ``baseline``."""
    results = []
    if df.empty:
        return pd.DataFrame()

    keys = ["eqn_name", "seed"]
    base_df = df[df["variant"] == baseline].set_index(keys)
    for variant in df["variant"].unique():
        if variant == baseline:
            continue
        comp_df = df[df["variant"] == variant].set_index(keys)
        common = base_df.index.intersection(comp_df.index)
        if len(common) == 0:
            continue
        for metric in metrics:
            a = base_df.loc[common][metric]
            b = comp_df.loc[common][metric]
            t_stat, p_val = stats.ttest_rel(a, b)
            results.append({
                "variant": variant,
                "metric": metric,
                "t_stat": t_stat,
                "p_value": p_val,
                "n": len(common),
            })
    return pd.DataFrame(results)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3e") -> str:
    """Return DataFrame as a GitHub style Markdown table."""
    if df.empty:
        return "(no data)"
    return df.to_markdown(index=False, floatfmt=floatfmt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results_dir", type=str, default="_results")
    parser.add_argument("--baseline", type=str, default="A1", help="variant used as baseline")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["l2_rel", "total_time", "peak_gpu_mem", "num_params"],
        help="metrics to summarize and compare",
    )
    parser.add_argument("--plot", action="store_true", help="create scatter plots")
    args = parser.parse_args()

    df = aggregate_results(args.results_dir)
    if df.empty:
        print("No runs found")
        exit()
    print(df.columns)
    summary = summarize_variants(df, args.metrics)
    print("\n## Summary")
    print(markdown_table(summary))

    ttest_df = paired_ttests(df, args.baseline, args.metrics)
    if not ttest_df.empty:
        print(f"\n## Paired t-tests vs {args.baseline}")
        print(markdown_table(ttest_df))

    if args.plot:
        xcol, ycol = args.metrics[0], args.metrics[1] if len(args.metrics) > 1 else args.metrics[0]
        plot_scatter(df, xcol, ycol, out_file=Path(args.results_dir) / "aggregate.png")
        for metric in args.metrics: 
            plot_boxplot(df, metric, out_file=Path(args.results_dir) / f"aggregate_bw_{metric}.png")
        
