from __future__ import annotations
from enum import Enum
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from typing_extensions import Final


class Cell:
    def __init__(self, round_to: int):
        self.round_to = round_to

    def __call__(self, data):
        mean = np.mean(data)
        if math.isnan(mean):
            return "N/A"
        std = np.std(data)
        round_level = self.round_to if std > 2 * pow(10, -self.round_to) else self.round_to + 1
        return f"{round(mean, round_level)} $\\pm$ {round(std, round_level)}"


def generate_table(
    df: pd.DataFrame,
    base_cols: list[str],
    metrics: list[str],
    metrics_renames: dict[str, str] | None = None,
    round_to: int = 2,
) -> pd.DataFrame:
    col_renames = {"data": "type", "method": "classifier"}
    if metrics_renames is not None:
        col_renames.update(metrics_renames)
    df = df[base_cols + metrics]
    df = df.rename(columns=col_renames, inplace=False)
    return (
        df.groupby(base_cols, sort=False)
        .agg(Cell(round_to=round_to))
        .reset_index(level=base_cols, inplace=False)
    )


class Metrics(Enum):
    acc = "accuracy"
    # ratios
    ar = "ar"
    tpr = "tpr"
    tnr = "tnr"
    # cluster metrics
    clust_acc = "clust_acc"
    clust_ari = "clust_ari"
    clust_nmi = "clust_nmi"


METRICS_COL_NAMES: Final = {
    Metrics.acc: lambda s, cl: f"Accuracy ({cl})",
    Metrics.ar: lambda s, cl: f"prob_pos_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tpr: lambda s, cl: f"TPR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tnr: lambda s, cl: f"TNR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.clust_acc: lambda s, cl: f"Clust/Context Accuracy",
    Metrics.clust_ari: lambda s, cl: f"Clust/Context ARI",
    Metrics.clust_nmi: lambda s, cl: f"Clust/Context NMI",
}
METRICS_RENAMES: Final = {
    Metrics.clust_acc: "Cluster. Acc. $\\uparrow$",
    Metrics.acc: "Acc. $\\uparrow$",
    Metrics.ar: "AR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tpr: "TPR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tnr: "TNR ratio $\\rightarrow 1.0 \\leftarrow$",
}
DEFAULT_METRICS: Final = [
    Metrics.clust_acc.value,
    Metrics.acc.value,
    Metrics.ar.value,
    Metrics.tpr.value,
    Metrics.tnr.value,
]


def main(
    csv_file: Path,
    metrics: List[Metrics] = typer.Option(default=DEFAULT_METRICS),
    sens_attr: str = typer.Option(default="colour"),
    classifier: str = typer.Option(default="pytorch_classifier"),
):
    df = pd.read_csv(csv_file)
    metrics_str = [METRICS_COL_NAMES[metric](sens_attr, classifier) for metric in metrics]
    metrics_renames = {
        metric_str: METRICS_RENAMES[metric] for metric_str, metric in zip(metrics_str, metrics)
    }
    # print(f"Using metrics: {metrics_str}")
    # print("------------------------------")
    tab = generate_table(
        df=df,
        base_cols=["misc.log_method"],  # first columns in the table
        metrics=metrics_str,
        metrics_renames=metrics_renames,
    )

    print(tab.to_latex(escape=False, index=False))


if __name__ == "__main__":
    typer.run(main)
