import os
import json
import pandas as pd
# pd.set_option('display.max_columns', 500)
# make print width larger
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt

def load_all_results(results_dir="_results"):
    """
    Walks through each subfolder in `results_dir`, loads `args.json` and `results.json`,
    merges them into one record per run, and returns a DataFrame of all runs.
    """
    records = []
    for run_folder in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_folder)
        if not os.path.isdir(run_path):
            continue

        args_file    = os.path.join(run_path, "args.json")
        results_file = os.path.join(run_path, "final_eval_results.json")  # or "final_results.json"

        # skip if either file is missing
        if not (os.path.exists(args_file) and os.path.exists(results_file)):
            continue

        with open(args_file, "r") as f:
            args = json.load(f)
        with open(results_file, "r") as f:
            results = json.load(f)

        # flatten and prefix to avoid name collisions
        record = {"run_folder": run_folder}
        record.update({f"arg_{k}": v for k, v in args.items()})
        record.update({f"res_{k}": v for k, v in results.items()})

        records.append(record)

    df = pd.DataFrame(records)
    return df

# Example usage:
df = load_all_results("_results").sort_values(by='res_l2_rel', ascending=True).reset_index(drop=True)
print(df)
print(df.iloc[0])

def plot_metric_boxplot(df, metric_col='res_final_loss', group_col='arg_run_name'):
    """
    Creates a box plot for the specified metric_col, with one box per unique group in group_col.
    Each box aggregates values across different seeds.
    """
    # ensure group column exists
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame")

    # prepare data
    unique_groups = df[group_col].unique()
    data = [df[df[group_col] == grp][metric_col].values for grp in unique_groups]

    # plot
    plt.figure()
    plt.boxplot(data, labels=unique_groups)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric_col)
    plt.title(f'Boxplot of {metric_col} across seeds for each run')
    # plt.tight_layout()
    plt.yscale('log') 
    plt.show()

plot_metric_boxplot(df, metric_col='res_l2_rel', group_col='arg_run_name')
