"""Utilities to aggregate experiment outputs and analyze ablations."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable, Sequence

import math
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
    sns.boxplot(data=df.sort_values(by="variant"), x="eqn_name", y=metric_col, hue="variant")
    plt.yscale('log')
    plt.xticks(rotation=45, ha="right")
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


def add_grad_method_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``grad_method`` column combining AD mode and STDE usage.

    The dataset stores ``ad_mode`` (``forward`` or ``reverse``) and whether
    stochastic diagonal estimators (STDE) were used via ``hess_diag_method`` and
    ``no_stde``.  This helper creates a single categorical column that is useful
    for grouping results by the effective gradient computation method.
    """

    if "grad_method" in df.columns:
        return df

    def _method(row: pd.Series) -> str:
        mode = row.get("ad_mode", "")
        hess = str(row.get("hess_diag_method", ""))
        no_stde = row.get("no_stde", True)
        if mode == "forward":
            if not no_stde and "stde" in hess:
                return "forward_stde"
            return "forward"
        return mode or "unknown"

    df["grad_method"] = df.apply(_method, axis=1)
    return df


def plot_metric_vs_num_params_by_eqn(
    df: pd.DataFrame,
    metric_col: str = "l2_rel",
    param_col: str = "num_params",
    out_file: str | Path = "metric_vs_params_by_eqn.png",
    log_axes: bool = True,
) -> None:
    """Plot ``metric_col`` against ``param_col`` for each equation.

    A line is drawn for each variant (model architecture) and subplots are
    arranged in a grid, one per equation with shared axes.  Axes are switched to
    logarithmic scale when values span multiple orders of magnitude.
    """

    if df.empty:
        raise ValueError("DataFrame is empty")

    eqns = sorted(df["eqn_name"].unique())
    n_eq = len(eqns)
    n_cols = math.ceil(math.sqrt(n_eq))
    n_rows = math.ceil(n_eq / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, eqn in zip(axes, eqns):
        sub = df[df["eqn_name"] == eqn]
        for variant, grp in sub.groupby("variant"):
            grp = grp.sort_values(param_col)
            ax.plot(grp[param_col], grp[metric_col], marker="o", label=variant)
        ax.set_title(eqn)
        ax.set_xlabel(param_col)
        ax.set_ylabel(metric_col)

    for ax in axes[n_eq:]:
        ax.remove()

    if log_axes:
        x_vals = df[param_col]
        y_vals = df[metric_col]
        if x_vals.max() / max(x_vals.min(), 1e-12) > 100:
            for ax in axes[:n_eq]:
                ax.set_xscale("log")
        if y_vals.max() / max(y_vals.min(), 1e-12) > 100:
            for ax in axes[:n_eq]:
                ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_file)
    print(f"Plot saved to {out_file}")


def plot_dims_vs_num_params_by_eqn(
    df: pd.DataFrame,
    dim_col: str = "hidden_features",
    param_col: str = "num_params",
    out_file: str | Path = "dims_vs_params_by_eqn.png",
    log_axes: bool = True,
) -> None:
    """Plot ``dim_col`` against ``param_col`` for each equation.

    The visualization mirrors :func:`plot_metric_vs_num_params_by_eqn` but uses
    ``dim_col`` (e.g. the model width) on the y-axis to illustrate how parameter
    counts scale with model dimensionality.
    """

    if df.empty:
        raise ValueError("DataFrame is empty")

    eqns = sorted(df["eqn_name"].unique())
    n_eq = len(eqns)
    n_cols = math.ceil(math.sqrt(n_eq))
    n_rows = math.ceil(n_eq / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, eqn in zip(axes, eqns):
        sub = df[df["eqn_name"] == eqn]
        for variant, grp in sub.groupby("variant"):
            grp = grp.sort_values(param_col)
            ax.plot(grp[param_col], grp[dim_col], marker="o", label=variant)
        ax.set_title(eqn)
        ax.set_xlabel(param_col)
        ax.set_ylabel(dim_col)

    for ax in axes[n_eq:]:
        ax.remove()

    if log_axes:
        x_vals = df[param_col]
        y_vals = df[dim_col]
        if x_vals.max() / max(x_vals.min(), 1e-12) > 100:
            for ax in axes[:n_eq]:
                ax.set_xscale("log")
        if y_vals.max() / max(y_vals.min(), 1e-12) > 100:
            for ax in axes[:n_eq]:
                ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_file)
    print(f"Plot saved to {out_file}")


def plot_metrics_by_grad_method(
    df: pd.DataFrame,
    metrics: Iterable[str] = ("l2_rel", "time_per_epoch", "peak_gpu_mem", "num_params"),
    out_file: str | Path = "metrics_by_grad_method.png",
) -> None:
    """Visualize ``metrics`` across gradient computation methods.

    Creates a grid of box plots, one for each metric, grouped by the derived
    ``grad_method`` column (see :func:`add_grad_method_column`).
    """

    if df.empty:
        raise ValueError("DataFrame is empty")

    add_grad_method_column(df)

    metrics = list(metrics)
    n = len(metrics)
    n_cols = min(2, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x="grad_method", y=metric, ax=ax)
        ax.set_xlabel("grad_method")
        ax.set_ylabel(metric)
        if df[metric].max() / max(df[metric].min(), 1e-12) > 100:
            ax.set_yscale("log")

    for ax in axes[n:]:
        ax.remove()

    fig.tight_layout()
    fig.savefig(out_file)
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
        default=["l2_rel", "time_per_epoch", "peak_gpu_mem", "num_params"],
        help="metrics to summarize and compare",
    )
    parser.add_argument("--plot", action="store_true", help="create scatter plots and exploratory figures")
    args = parser.parse_args()

    df = aggregate_results(args.results_dir)
    if df.empty:
        print("No runs found")
        exit()
    print(df.columns)
    summary = summarize_variants(df, args.metrics)
    print("\n## Summary")
    print(markdown_table(summary))

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "aggregate_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # ttest_df = paired_ttests(df, args.baseline, args.metrics)
    # if not ttest_df.empty:
    #     print(f"\n## Paired t-tests vs {args.baseline}")
    #     print(markdown_table(ttest_df))
    #     ttest_df.to_csv(out_dir / "paired_ttests.csv", index=False)

    if args.plot:
        xcol, ycol = args.metrics[0], args.metrics[1] if len(args.metrics) > 1 else args.metrics[0]
        plot_scatter(df, xcol, ycol, out_file=Path(args.results_dir) / "aggregate.png")
        for metric in args.metrics:
            plot_boxplot(df, metric, out_file=Path(args.results_dir) / f"aggregate_bw_{metric}.png")
        plot_metric_vs_num_params_by_eqn(
            df,
            metric_col="l2_rel",
            out_file=Path(args.results_dir) / "l2_vs_params_by_eqn.png",
        )
        plot_dims_vs_num_params_by_eqn(
            df,
            dim_col="hidden_features",
            out_file=Path(args.results_dir) / "dims_vs_params_by_eqn.png",
        )
        plot_metrics_by_grad_method(
            df,
            metrics=["l2_rel", "time_per_epoch", "peak_gpu_mem", "num_params"],
            out_file=Path(args.results_dir) / "metrics_by_grad_method.png",
        )
        
