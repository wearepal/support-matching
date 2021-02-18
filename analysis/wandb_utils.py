from __future__ import annotations
from enum import Enum, auto
import math
from pathlib import Path
from typing import Callable, List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing_extensions import Final

__all__ = ["Metrics", "load_data", "plot"]


class Metrics(Enum):
    acc = auto()
    hgr = auto()  # Renyi correlation
    # ratios
    prr = auto()
    tprr = auto()
    tnrr = auto()
    # diffs
    prd = auto()
    tprd = auto()
    tnrd = auto()
    # cluster metrics
    clust_acc = auto()
    clust_ari = auto()
    clust_nmi = auto()


class Aggregation(Enum):
    none = auto()
    min = auto()
    max = auto()


METRICS_COL_NAMES: Final = {
    Metrics.acc: lambda s, cl: f"Accuracy ({cl})",
    Metrics.hgr: lambda s, cl: f"Renyi preds and s ({cl})",
    Metrics.prr: lambda s, cl: f"prob_pos_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tprr: lambda s, cl: f"TPR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.tnrr: lambda s, cl: f"TNR_{s}_0.0÷{s}_1.0 ({cl})",
    Metrics.clust_acc: lambda s, cl: f"Clust/Context Accuracy",
    Metrics.clust_ari: lambda s, cl: f"Clust/Context ARI",
    Metrics.clust_nmi: lambda s, cl: f"Clust/Context NMI",
}

AGG_METRICS_COL_NAMES: Final = {
    Metrics.prr: lambda s, cl: (
        f"prob_pos_{s}_0.0÷{s}_1.0 ({cl})",
        f"prob_pos_{s}_0.0÷{s}_2.0 ({cl})",
        f"prob_pos_{s}_1.0÷{s}_2.0 ({cl})",
    ),
    Metrics.tprr: lambda s, cl: (
        f"TPR_{s}_0.0÷{s}_1.0 ({cl})",
        f"TPR_{s}_0.0÷{s}_2.0 ({cl})",
        f"TPR_{s}_1.0÷{s}_2.0 ({cl})",
    ),
    Metrics.tnrr: lambda s, cl: (
        f"TNR_{s}_0.0÷{s}_1.0 ({cl})",
        f"TNR_{s}_0.0÷{s}_2.0 ({cl})",
        f"TNR_{s}_1.0÷{s}_2.0 ({cl})",
    ),
    Metrics.prd: lambda s, cl: (
        f"prob_pos_{s}_0.0-{s}_1.0 ({cl})",
        f"prob_pos_{s}_0.0-{s}_2.0 ({cl})",
        f"prob_pos_{s}_1.0-{s}_2.0 ({cl})",
    ),
    Metrics.tprd: lambda s, cl: (
        f"TPR_{s}_0.0-{s}_1.0 ({cl})",
        f"TPR_{s}_0.0-{s}_2.0 ({cl})",
        f"TPR_{s}_1.0-{s}_2.0 ({cl})",
    ),
    Metrics.tnrd: lambda s, cl: (
        f"TNR_{s}_0.0-{s}_1.0 ({cl})",
        f"TNR_{s}_0.0-{s}_2.0 ({cl})",
        f"TNR_{s}_1.0-{s}_2.0 ({cl})",
    ),
}

METRICS_RENAMES: Final = {
    Metrics.clust_acc: lambda a: f"Cluster. Acc.{a} $\\rightarrow$",
    Metrics.acc: lambda a: f"Accuracy{a} $\\rightarrow$",
    Metrics.hgr: lambda a: f"$\\leftarrow$ HGR{a}",
    Metrics.prr: lambda a: f"PR ratio{a} $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tprr: lambda a: f"TPR ratio{a} $\\rightarrow 1.0 \\leftarrow$",
    Metrics.tnrr: lambda a: f"TNR ratio{a} $\\rightarrow 1.0 \\leftarrow$",
    Metrics.prd: lambda a: f"PR diff{a} $\\rightarrow 0.0 \\leftarrow$",
    Metrics.tprd: lambda a: f"TPR diff{a} $\\rightarrow 0.0 \\leftarrow$",
    Metrics.tnrd: lambda a: f"TNR diff{a} $\\rightarrow 0.0 \\leftarrow$",
}

METHOD_RENAMES: Final = {
    "ranking-fdm": "Ranking",
    "ranking-fdm-4": "Ranking",
    "ranking-fdm-8": "Ranking (8)",
    "no-cluster-fdm": "No bal.",
    "perfect-cluster": "Perfect",
    "kmeans-fdm": "k-means",
    "kmeans-fdm-8": "k-means (8)",
    # "baseline_cnn": "K&C",
    "baseline_cnn": "ERM",
    "baseline_erm": "ERM",
    "baseline_oracle": "ERM (LD)",
    "baseline_dro_0.01": "DRO",
    "baseline_dro_0.1": "DRO",
    "baseline_dro_0.3": "DRO",
    "baseline_dro_1.0": "DRO",
    "baseline_dro": "DRO",
    "baseline_gdro": "gDRO",
    "baseline_lff": "LfF",
}

KNOWN_CLASSIFIERS: Final = ["pytorch_classifier", "cnn", "dro", "gdro", "lff", "erm", "oracle"]


def merge_cols(df, correct_col: str, incorrect_col: str) -> bool:
    try:
        to_merge = df[incorrect_col]
    except KeyError:
        return False
    df[correct_col] = df[correct_col].combine_first(to_merge)
    return True


def compute_min(df: pd.DataFrame, to_aggregate: tuple[str], rename: Callable[[str], str]) -> str:
    ratios = tuple(df[col] for col in to_aggregate)
    min_ = pd.Series(1, ratios[0].index)
    for ratio in ratios:
        min_ = min_.where(min_ < ratio, ratio)
    new_col = rename(" min")
    df[new_col] = min_
    return new_col


def compute_max(df: pd.DataFrame, to_aggregate: tuple[str], rename: Callable[[str], str]) -> str:
    diffs = tuple(df[col] for col in to_aggregate)
    max_ = pd.Series(0, diffs[0].index)
    for diff in diffs:
        max_ = max_.where(max_ > diff, diff)
    new_col = rename(" max")
    df[new_col] = max_
    return new_col


def simple_concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, axis="index", sort=False, ignore_index=True)


def load_data(*csv_files: Path) -> pd.DataFrame:
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))
    return simple_concat(*dfs)


def plot(
    data: pd.DataFrame,
    groupby: str = "misc.log_method",
    metrics: List[Metrics] = [Metrics.acc],
    sens_attr: str = "colour",
    output_dir: Path = Path("."),
    file_format: str = "png",
    file_prefix: str = "",
    fig_dim: Tuple[float, float] = (4.0, 6.0),
    y_limits: Tuple[float, float] = (math.nan, math.nan),
    x_limits: Tuple[float, float] = (math.nan, math.nan),
    agg: Aggregation = Aggregation.none,
) -> None:
    df = data.copy()

    for metric in metrics:
        if agg is Aggregation.none:
            column_to_plot = METRICS_COL_NAMES[metric](sens_attr, KNOWN_CLASSIFIERS[0])
            col_renames = {column_to_plot: METRICS_RENAMES[metric]("")}

            # merge all other classifier-based columns into the first column
            for classifier in KNOWN_CLASSIFIERS[1:]:
                merge_cols(df, column_to_plot, METRICS_COL_NAMES[metric](sens_attr, classifier))
        else:
            cols_to_aggregate = AGG_METRICS_COL_NAMES[metric](sens_attr, KNOWN_CLASSIFIERS[0])

            # merge all other classifier-based columns into the first column
            for classifier in KNOWN_CLASSIFIERS[1:]:
                for col_to_aggregate, variant in zip(
                    cols_to_aggregate, AGG_METRICS_COL_NAMES[metric](sens_attr, classifier)
                ):
                    merge_cols(df, col_to_aggregate, variant)

            if agg is Aggregation.max:
                column_to_plot = compute_max(df, cols_to_aggregate, METRICS_RENAMES[metric])
            else:
                column_to_plot = compute_min(df, cols_to_aggregate, METRICS_RENAMES[metric])

            # no need for a rename because we wrote the result in the correctly named column
            col_renames = {column_to_plot: column_to_plot}

        base_cols = [groupby]
        col_renames[groupby] = "Method"

        df = df[base_cols + [column_to_plot]]
        df = df.rename(columns=col_renames, inplace=False)
        df = df.replace({"Method": METHOD_RENAMES}, inplace=False)

        filename = metric.name
        if agg is not Aggregation.none:
            filename += f"-{agg.name}"
        filename += f".{file_format}"
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        # sns.set_style("whitegrid")
        fig, plot = plt.subplots(figsize=fig_dim, dpi=300, facecolor="white")
        sns.boxplot(y="Method", x=col_renames[column_to_plot], data=df, ax=plot, whis=3.0)
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
