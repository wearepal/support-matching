from __future__ import annotations
from enum import Enum, auto
import math
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing_extensions import Final

__all__ = ["Metrics", "load_data", "plot"]


class Metrics(Enum):
    acc = auto()
    hgr = auto()  # Renyi correlation
    # ratios
    pr = auto()
    tpr = auto()
    tnr = auto()
    # cluster metrics
    clust_acc = auto()
    clust_ari = auto()
    clust_nmi = auto()


METRICS_COL_NAMES: Final = {
    Metrics.acc: lambda s, cl: f"Accuracy ({cl})",
    Metrics.hgr: lambda s, cl: f"Renyi preds and s ({cl})",
    Metrics.pr: lambda s, cl: f"prob_pos_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tpr: lambda s, cl: f"TPR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tnr: lambda s, cl: f"TNR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.clust_acc: lambda s, cl: f"Clust/Context Accuracy",
    Metrics.clust_ari: lambda s, cl: f"Clust/Context ARI",
    Metrics.clust_nmi: lambda s, cl: f"Clust/Context NMI",
}

METRICS_RENAMES: Final = {
    Metrics.clust_acc: "Cluster. Acc. $\\rightarrow$",
    Metrics.acc: "Accuracy $\\rightarrow$",
    Metrics.hgr: "HGR $\\leftarrow$",
    Metrics.pr: "PR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tpr: "TPR ratio $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tnr: "TNR ratio $\\rightarrow 1.0 \\leftarrow$",
}

METHOD_RENAMES: Final = {
    "ranking-fdm": "Ranking",
    "no-cluster-fdm": "No bal.",
    "perfect-cluster": "Perfect",
    "kmeans-fdm": "k-means",
    # "baseline_cnn": "K&C",
    "baseline_cnn": "ERM",
    "baseline_dro_0.01": "DRO",
    "baseline_dro_0.1": "DRO",
    "baseline_dro_0.3": "DRO",
    "baseline_dro_1.0": "DRO",
    "baseline_dro": "DRO",
    "baseline_gdro": "gDRO",
    "baseline_lff": "LfF",
}

KNOWN_CLASSIFIERS: Final = ["pytorch_classifier", "cnn", "dro", "gdro", "lff"]


def merge_cols(df, correct_col: str, incorrect_col: str) -> bool:
    try:
        to_merge = df[incorrect_col]
    except KeyError:
        return False
    df[correct_col] = df[correct_col].combine_first(to_merge)
    return True


def load_data(*csv_files: Path) -> pd.DataFrame:
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))
    return pd.concat(dfs, axis="index", sort=False, ignore_index=True)


def plot(
    data: pd.DataFrame,
    groupby: str = "misc.log_method",
    metrics: List[Metrics] = [Metrics.acc],
    sens_attr: str = "colour",
    output_dir: Path = Path("."),
    file_format: str = "png",
    file_prefix: str = "",
    fig_dim: Tuple[float, float] = (4.0, 6.0),
    y_limits: Tuple[float, float] = (float("nan"), float("nan")),
    x_limits: Tuple[float, float] = (float("nan"), float("nan")),
) -> None:
    df = data.copy()
    classifier = KNOWN_CLASSIFIERS[0]

    for metric in metrics:
        metric_str = METRICS_COL_NAMES[metric](sens_attr, KNOWN_CLASSIFIERS[0])
        col_renames = {metric_str: METRICS_RENAMES[metric]}

        # merge all other classifier-based columns into the first column
        for classifier in KNOWN_CLASSIFIERS[1:]:
            merge_cols(df, metric_str, METRICS_COL_NAMES[metric](sens_attr, classifier))

        base_cols = [groupby]
        col_renames[groupby] = "Method"

        df = df[base_cols + [metric_str]]
        df = df.rename(columns=col_renames, inplace=False)
        df = df.replace({"Method": METHOD_RENAMES}, inplace=False)

        filename = f"{metric.name}.{file_format}"
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        # sns.set_style("whitegrid")
        fig, plot = plt.subplots(figsize=fig_dim, dpi=300, facecolor="white")
        sns.boxplot(y="Method", x=col_renames[metric_str], data=df, ax=plot, whis=3.0)
        hatches = ["/", "\\", ".", "x", "/", "\\", ".", "x"]
        for hatch, patch in zip(hatches, plot.artists):
            # patch.set_hatch(hatch)
            patch.set_edgecolor("black")
            # patch.set_facecolor("lightgrey")

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
