"""Utility to aggregate experiment outputs and visualize error metrics."""

import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def aggregate_results(results_dir="_results"):
    """Load config and evaluation results from each run into a DataFrame."""
    records = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results directory '{results_dir}' not found")

    for run in sorted(os.listdir(results_dir)):
        run_path = os.path.join(results_dir, run)
        if not os.path.isdir(run_path):
            continue

        cfg_path = os.path.join(run_path, "config.json")
        res_path = os.path.join(run_path, "final_eval_results.json")
        if not (os.path.exists(cfg_path) and os.path.exists(res_path)):
            continue

        with open(cfg_path) as f:
            config = json.load(f)
        with open(res_path) as f:
            results = json.load(f)

        record = {"run": run}
        record.update(config)
        record.update(results)
        records.append(record)

    return pd.DataFrame(records)


def plot_scatter(df, xcol, ycol, out_file="aggregate_plot.png"):
    """Create an annotated scatter plot of l2_rel vs l1_rel, colored by eqn_name."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=xcol, y=ycol, hue="eqn_name")
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


if __name__ == "__main__":
    df = aggregate_results()
    print(df)
    print(df.columns)
    if not df.empty:
        plot_scatter(df, "l1_rel", "final_loss", out_file="l1_floss.png")
        
