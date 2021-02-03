from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from typing_extensions import Final

from wandb_csv_to_table import METRICS_COL_NAMES, Metrics


METRICS_RENAMES: Final = {
    Metrics.clust_acc: "Cluster. Acc. $\\rightarrow$",
    Metrics.acc: "Accuracy $\\rightarrow$",
    Metrics.ar: "AR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tpr: "TPR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tnr: "TNR ratio $\\rightarrow 1.0 \\leftarrow$",
}


def merge_cols(df, correct_col: str, incorrect_col: str):
    df[correct_col] = df[correct_col].combine_first(df[incorrect_col])


def main(
    csv_files: List[Path],
    groupby: str = typer.Option("misc.log_method", "--groupby", "-g"),
    metrics: List[Metrics] = typer.Option([Metrics.acc.value], "--metrics", "-m"),
    sens_attr: str = typer.Option("colour", "--sens-attr", "-s"),
    classifiers: List[str] = typer.Option("pytorch_classifier", "--classifiers", "-c"),
    output_dir: Path = typer.Option(Path("."), "--output", "-o"),
    file_format: str = typer.Option("png", "--file-format", "-f"),
    file_prefix: str = typer.Option("", "--file_prefix", "-p"),
    fig_dim: Tuple[float, float] = typer.Option((4.0, 6.0), "--fig-dim", "-d"),
    y_limits: Tuple[float, float] = typer.Option((float("nan"), float("nan")), "--y-limits", "-y"),
    x_limits: Tuple[float, float] = typer.Option((float("nan"), float("nan")), "--x-limits", "-x"),
) -> None:
    print("---------------------------------------")
    print("Settings:")
    print(f"    metrics (-m)    :  [{', '.join(metric.value for metric in metrics)}]")
    print(f'    classifiers (-c):  "{classifiers}"')
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
    classifier = classifiers[0]

    for metric in metrics:
        metric_str = METRICS_COL_NAMES[metric](sens_attr, classifiers[0])
        col_renames = {metric_str: METRICS_RENAMES[metric]}

        # merge all other classifier-based columns into the first column
        for classifier in classifiers[1:]:
            merge_cols(df, metric_str, METRICS_COL_NAMES[metric](sens_attr, classifier))

        base_cols = [groupby]
        col_renames[groupby] = "Method"
        method_renames = {
            "ranking-fdm": "Ranking",
            "no-cluster-fdm": "No bal.",
            "perfect-cluster": "Perfect",
            "kmeans-fdm": "k-means",
            # "baseline_cnn": "K&C",
            "baseline_cnn": "ERM",
            "baseline_dro_0.1": "DRO",
        }

        df = df[base_cols + [metric_str]]
        df = df.rename(columns=col_renames, inplace=False)
        df = df.replace({"Method": method_renames}, inplace=False)

        filename = f"{metric.value}.{file_format}"
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        # sns.set_style("whitegrid")
        fig, plot = plt.subplots(figsize=fig_dim, dpi=300)
        sns.boxplot(y="Method", x=col_renames[metric_str], data=df, ax=plot, whis=3.0)
        hatches = ["/", "\\", ".", "x", "/", "\\", ".", "x"]
        for hatch, patch in zip(hatches, plot.artists):
            # patch.set_hatch(hatch)
            patch.set_edgecolor('black')
            patch.set_facecolor('lightgrey')

        # if you only want to set one ylim, then pass "nan" on the commendline for the other value
        plot.set_ylim(
            ymin=y_limits[0] if not math.isnan(y_limits[0]) else None,
            ymax=y_limits[1] if not math.isnan(y_limits[1]) else None,
        )
        plot.set_xlim(
            xmin=x_limits[0] if not math.isnan(x_limits[0]) else None,
            xmax=x_limits[1] if not math.isnan(x_limits[1]) else None,
        )
        # plot.grid(axis="y")
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
