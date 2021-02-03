from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from wandb_csv_to_table import Metrics, METRICS_COL_NAMES, METRICS_RENAMES


def main(
    csv_files: List[Path],
    groupby: str = typer.Option("misc.log_method", "--groupby", "-g"),
    metric: Metrics = typer.Option(Metrics.acc.value, "--metric", "-m"),
    sens_attr: str = typer.Option("colour", "--sens-attr", "-s"),
    classifier: str = typer.Option("pytorch_classifier", "--classifiers", "-c"),
    output_dir: Path = typer.Option(Path("."), "--output", "-o"),
    file_format: str = typer.Option("png", "--file-format", "-f"),
    file_prefix: str = typer.Option("", "--file_prefix", "-p"),
    fig_dim: Tuple[float, float] = typer.Option((4.0, 6.0), "--fig-dim", "-d"),
) -> None:
    print("---------------------------------------")
    print("Settings:")
    print(f"    metric (-m)     :  {metric.value}")
    print(f'    classifier (-c) :  "{classifier}"')
    print(f'    file_format (-f):  "{file_format}"')
    print(f'    file_prefix (-p):  "{file_prefix}"')
    print(f'    groupby (-g)    :  "{groupby}"')
    print(f'    output_dir (-o) :  "{output_dir}"')
    print(f'    sens_attr (-s)  :  "{sens_attr}"')
    print(f"    fig_dim (-f)    :  {fig_dim}")
    print("---------------------------------------\n")
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))
    df = pd.concat(dfs, axis="index", sort=False, ignore_index=True)

    metric_str = METRICS_COL_NAMES[metric](sens_attr, classifier)
    col_renames = {metric_str: METRICS_RENAMES[metric]}

    base_cols = [groupby]
    col_renames[groupby] = "Method"
    method_renames = {
        "ranking-fdm": "Ranking",
        "no-cluster-fdm": "No bal.",
        "perfect-cluster": "Ground truth",
        "kmeans-fdm": "k-means",
    }

    df = df[base_cols + [metric_str]]
    df = df.rename(columns=col_renames, inplace=False)
    df = df.replace({"Method": method_renames}, inplace=False)

    filename = f"{metric.value}.{file_format}"
    if file_prefix:
        filename = f"{file_prefix}_{filename}"
    fig, plot = plt.subplots(figsize=fig_dim, dpi=300)
    sns.boxplot(x="Method", y=col_renames[metric_str], data=df, ax=plot, whis=3.0)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
