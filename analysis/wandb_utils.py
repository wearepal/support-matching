from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
from typing import Final, NamedTuple, Optional, TypedDict, TypeVar, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from ranzen.wandb import RunsDownloader
import seaborn as sns

__all__ = [
    "Group",
    "MethodName",
    "Metrics",
    "PlotKwargs",
    "PlotStyle",
    "concat_with_suffix",
    "download_groups",
    "load_data",
    "plot",
    "simple_concat",
]


class Metrics(Enum):
    # each metric an optionally take the senstitive attribute as the parameter {s}
    # if there is aggregation, then the name will be included as the parameter {a}
    acc = ("Accuracy", "Accuracy{a} $\\rightarrow$")
    rob_acc = ("Robust_Accuracy", "Robust accuracy{a} $\\rightarrow$")
    rob_tpr = ("Robust_TPR", "Robust TPR{a} $\\rightarrow$")
    rob_tpr_ovr = ("Robust OvR TPR", "Robust TPR one versus rest $\\rightarrow$")
    hgr = ("Renyi preds and s", "$\\leftarrow$ HGR{a}")
    # ratios
    prr = ("prob_pos_{s}_0.0÷{s}_1.0", "PR ratio{a} $\\rightarrow 1.0 \\leftarrow$")
    tprr = ("TPR_{s}_0.0÷{s}_1.0", "TPR ratio{a} $\\rightarrow 1.0 \\leftarrow$")
    tnrr = ("TNR_{s}_0.0÷{s}_1.0", "TNR ratio{a} $\\rightarrow 1.0 \\leftarrow$")
    # diffs
    prd = ("", "PR diff{a} $\\rightarrow 0.0 \\leftarrow$")
    tprd = ("", "TPR diff{a} $\\rightarrow 0.0 \\leftarrow$")
    tnrd = ("", "TNR diff{a} $\\rightarrow 0.0 \\leftarrow$")
    # cluster metrics
    clust_acc = ("Clust/Context Accuracy", "Cluster. Acc.{a} $\\rightarrow$")
    clust_ari = ("Clust/Context ARI", "")
    clust_nmi = ("Clust/Context NMI", "")

    def __init__(self, col_name: str, display_name: str):
        self.col_name = col_name
        self.display_name = display_name


class Aggregation(Enum):
    none = auto()
    min = auto()
    max = auto()


AGG_METRICS_COL_NAMES: Final = {
    Metrics.acc: lambda s: (f"Accuracy_{s}_0.0", f"Accuracy_{s}_1.0"),
    Metrics.prr: lambda s: (
        f"prob_pos_{s}_0.0÷{s}_1.0",
        f"prob_pos_{s}_0.0÷{s}_2.0",
        f"prob_pos_{s}_1.0÷{s}_2.0",
    ),
    Metrics.tprr: lambda s: (
        f"TPR_{s}_0.0÷{s}_1.0",
        f"TPR_{s}_0.0÷{s}_2.0",
        f"TPR_{s}_1.0÷{s}_2.0",
    ),
    Metrics.tnrr: lambda s: (
        f"TNR_{s}_0.0÷{s}_1.0",
        f"TNR_{s}_0.0÷{s}_2.0",
        f"TNR_{s}_1.0÷{s}_2.0",
    ),
    Metrics.prd: lambda s: (
        f"prob_pos_{s}_0.0-{s}_1.0",
        f"prob_pos_{s}_0.0-{s}_2.0",
        f"prob_pos_{s}_1.0-{s}_2.0",
    ),
    Metrics.tprd: lambda s: (
        f"TPR_{s}_0.0-{s}_1.0",
        f"TPR_{s}_0.0-{s}_2.0",
        f"TPR_{s}_1.0-{s}_2.0",
    ),
    Metrics.tnrd: lambda s: (
        f"TNR_{s}_0.0-{s}_1.0",
        f"TNR_{s}_0.0-{s}_2.0",
        f"TNR_{s}_1.0-{s}_2.0",
    ),
}


class MethodName(Enum):
    # ours_no_balancing = "Ours (No Balancing)"
    # ours_bag_oracle = "Ours (Bag Oracle)"
    ours_no_balancing = "Ours (Oracle=$\\varnothing$)"
    ours_bag_oracle = "Ours (Oracle=B)"
    ours_clustering = "Ours (Clustering)"
    erm = "ERM (Oracle=$\\varnothing$)"
    gdro = "gDRO (Oracle=$\\varnothing$)"
    gdro_oracle = "gDRO (Oracle=Y)"
    george = "GEORGE (Oracle=$\\varnothing$)"
    dro = "DRO (Oracle=V)"
    lff = "LfF (Oracle=$\\varnothing$)"
    erm_oracle = "ERM (Oracle=B & Y)"
    dfr = "DFR (Oracle=B & Y)"
    ours_no_bags = "Ours (instance-wise)"
    ours_with_bags = "Ours (bag-wise)"


class CustomMethod(NamedTuple):
    value: str


METHOD_RENAMES: Final = {
    "balanced-False": MethodName.ours_no_balancing.value,
    "balanced-True": MethodName.ours_bag_oracle.value,
    "balanced-with-clustering": "Ours (Clustering)",
    "baseline_cnn": MethodName.erm.value,
    "baseline_dro": MethodName.dro.value,
    "baseline_dro_0.01": MethodName.dro.value,
    "baseline_dro_0.1": MethodName.dro.value,
    "baseline_dro_0.3": MethodName.dro.value,
    "baseline_dro_1.0": MethodName.dro.value,
    "baseline_erm": MethodName.erm.value,
    "baseline_gdro": MethodName.gdro.value,
    "baseline_lff": MethodName.lff.value,
    "baseline_oracle": MethodName.erm_oracle.value,
    "cluster_and_gdro": MethodName.george.value,
    "erm_no_context_no_reg": MethodName.erm.value,
    "kmeans-fdm": "k-means",
    "kmeans-fdm-6": "k-means (6)",
    "kmeans-fdm-8": "k-means (8)",
    "no-cluster-fdm": "Ours (No Balancing)",
    "oracle_gdro": "gDRO (Label Oracle)",
    "perfect-cluster": "Ours (Bag Oracle)",
    "no-cluster-suds": "Ours (No Balancing)",
    "perfect-cluster-nomil": "Inst. (Bag Oracle)",
    "no-cluster-fdm-nomil": "Inst. (No Balancing)",
    "ranking-suds": "Ours (Clustering)",
    "ranking-fdm": "Ours (Clustering)",
    "ranking-fdm-4": "Ours (Clustering; k=4)",
    "ranking-fdm-6": "Ours (Clustering; k=6)",
    "ranking-fdm-8": "Ours (Clustering; k=8)",
    "ranking-fdm-nomil": "Inst. (Clustering)",
    "ss_ae": "AutoEncoder",
}

KNOWN_CLASSIFIERS: Final = [
    "pytorch_classifier",
    "cnn",
    "dro",
    "gdro",
    "lff",
    "erm",
    "oracle",
]


def merge_cols(df: pd.DataFrame, correct_col: str, incorrect_col: str) -> bool:
    try:
        to_merge = df[incorrect_col]
    except KeyError:
        return False
    df[correct_col] = df[correct_col].combine_first(to_merge)
    return True


@dataclass
class Aggregate:
    agg: Callable[[pd.Series, pd.Series], pd.Series]
    suffix: str
    default_value: int

    def __call__(self, df: pd.DataFrame, to_aggregate: list[str], display_name: str) -> str:
        ratios = tuple(df[col] for col in to_aggregate)
        min_ = pd.Series(self.default_value, ratios[0].index)
        for ratio in ratios:
            min_ = min_.where(self.agg(min_, ratio), ratio)
        new_col = display_name.format(a=self.suffix)
        df[new_col] = min_
        return new_col


compute_min = Aggregate(agg=lambda x, y: x < y, suffix=" min", default_value=1)
compute_max = Aggregate(agg=lambda x, y: x > y, suffix=" max", default_value=0)


def simple_concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, axis="index", sort=False, ignore_index=True)


def concat_with_suffix(
    dfs: dict[str, pd.DataFrame], groupby: str = "misc.log_method"
) -> pd.DataFrame:
    renamed_dfs: list[pd.DataFrame] = []
    for suffix, df in dfs.items():
        copy = df.copy()
        copy[groupby] += suffix
        renamed_dfs.append(copy)
    return pd.concat(renamed_dfs, axis="index", sort=False, ignore_index=True)


def load_data(*csv_files: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    return simple_concat(*dfs)


class Group(NamedTuple):
    name: MethodName | CustomMethod
    metrics_prefix: str = ""
    metrics_suffix: str = " (pytorch_classifier)"


def load_groups(group_mapping: dict[Path, Group]) -> pd.DataFrame:
    """Load data from CSV files.

    This method can also remove metric prefixes and suffixes.
    """
    return _gather_groups(group_mapping, pd.read_csv)


def download_groups(downloader: RunsDownloader, group_mapping: dict[str, Group]) -> pd.DataFrame:
    """Download groups from W&B which do not have `misc.log_method` set.

    This method can also remove metric prefixes and suffixes.
    """
    return _gather_groups(group_mapping, downloader.groups)


T = TypeVar("T")


def _gather_groups(
    group_mapping: dict[T, Group], retriever: Callable[[T], pd.DataFrame]
) -> pd.DataFrame:
    dfs = []
    for group, (method_name, metric_prefix, metric_suffix) in group_mapping.items():
        df = retriever(group)
        df = df.rename(
            columns={
                col: col.removeprefix(metric_prefix).removesuffix(metric_suffix)
                for col in df.columns
            },
            inplace=False,
        )
        df["misc.log_method"] = method_name.value
        dfs.append(df)
    return pd.concat(dfs, axis="index", sort=False, ignore_index=True)


class PlotStyle(Enum):
    boxplot = auto()
    boxplot_hue = auto()
    lineplot = auto()
    scatterplot = auto()


class PlotKwargs(TypedDict, total=False):
    file_format: str
    fig_dim: tuple[float, float]
    file_prefix: str
    sens_attr: str
    output_dir: Union[Path, str]


def plot(
    data: pd.DataFrame,
    groupby: str = "misc.log_method",
    metrics: list[Metrics] = [Metrics.acc],
    sens_attr: str = "colour",
    output_dir: Union[Path, str] = Path("."),
    file_format: str = "png",
    file_prefix: str = "",
    fig_dim: tuple[float, float] = (4.0, 6.0),
    y_limits: tuple[float, float] = (math.nan, math.nan),
    x_limits: tuple[float, float] = (math.nan, math.nan),
    agg: Aggregation = Aggregation.none,
    fillna: bool = False,
    hide_left_ticks: bool = False,
    x_label: Optional[str] = None,
    plot_style: PlotStyle = PlotStyle.boxplot,
    plot_title: Optional[str] = None,
    with_legend: bool = True,
) -> None:
    for metric in metrics:
        df = data.copy()
        df, renamed_col_to_plot = _prepare_dataframe(
            df,
            groupby=groupby,
            agg=agg,
            metric=metric,
            sens_attr=sens_attr,
            fillna=fillna,
        )

        fig = _make_plot(
            df=df,
            renamed_col_to_plot=renamed_col_to_plot,
            fig_dim=fig_dim,
            x_limits=x_limits,
            y_limits=y_limits,
            hide_left_ticks=hide_left_ticks,
            x_label=x_label,
            plot_style=plot_style,
            plot_title=plot_title,
            with_legend=with_legend,
        )
        filename = _prepare_filename(
            metric=metric, agg=agg, file_format=file_format, file_prefix=file_prefix
        )
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, bbox_inches="tight")


def _prepare_dataframe(
    df: pd.DataFrame,
    groupby: str,
    agg: Aggregation,
    metric: Metrics,
    sens_attr: str,
    fillna: bool,
) -> tuple[pd.DataFrame, str]:
    """Merge columns to get the right metrics and find out the right column to plot.

    The problem that this function solves is that we at some point decided to include the classifier
    name in the metric name. So, for example "Accuracy (pytorch_classifier)" or "Accuracy (Logistic
    Regression)". This function normalizes the metric names so that they're all the same and all in
    one column.
    """
    if agg is Aggregation.none:
        column_to_plot = metric.col_name.format(s=sens_attr)
        col_renames = {column_to_plot: metric.display_name.format(a="")}

        # # merge all other classifier-based columns into the first column
        # for classifier in KNOWN_CLASSIFIERS:
        #     merge_cols(df, column_to_plot, f"{metric.col_name.format(s=sens_attr)} ({classifier})")
    else:
        cols_to_aggregate = list(AGG_METRICS_COL_NAMES[metric](sens_attr))

        # # merge all other classifier-based columns into the first column
        # for classifier in KNOWN_CLASSIFIERS[1:]:
        #     suffixed_metrics = [
        #         f"{n} ({classifier})" for n in AGG_METRICS_COL_NAMES[metric](sens_attr)
        #     ]
        #     for col_to_aggregate, variant in zip(cols_to_aggregate, suffixed_metrics):
        #         merge_cols(df, col_to_aggregate, variant)

        if agg is Aggregation.max:
            column_to_plot = compute_max(df, cols_to_aggregate, metric.display_name)
        else:
            column_to_plot = compute_min(df, cols_to_aggregate, metric.display_name)

        # no need for a rename because we wrote the result in the correctly named column
        col_renames = {column_to_plot: column_to_plot}

    base_cols = [groupby]
    if groupby != "misc.log_method":
        base_cols.append("misc.log_method")
    col_renames[groupby] = "Method"

    assert "misc.log_method" in df, str(df.columns)
    df = df[base_cols + [column_to_plot]]
    df = df.rename(columns=col_renames, inplace=False)
    df = df.replace({"Method": METHOD_RENAMES}, inplace=False)
    if fillna:
        df = df.fillna(0, inplace=False).replace("NaN", 0, inplace=False)
    return df, col_renames[column_to_plot]


def _prepare_filename(metric: Metrics, agg: Aggregation, file_format: str, file_prefix: str) -> str:
    filename = metric.name
    if agg is not Aggregation.none:
        filename += f"-{agg.name}"
    filename += f".{file_format}"
    if file_prefix:
        filename = f"{file_prefix}_{filename}"
    return filename


def _make_plot(
    df: pd.DataFrame,
    renamed_col_to_plot: str,
    fig_dim: tuple[float, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    hide_left_ticks: bool,
    x_label: Optional[str],
    plot_style: PlotStyle,
    plot_title: Optional[str] = None,
    with_legend: bool = True,
) -> Figure:
    # sns.set_style("whitegrid")
    plot: Axes
    fig, plot = plt.subplots(figsize=fig_dim, dpi=300, facecolor="white")
    if plot_style is PlotStyle.boxplot:
        sns.set_palette("husl", 12)
        sns.boxplot(
            y="Method",
            x=renamed_col_to_plot,
            data=df,
            ax=plot,
            whis=1.0,
            medianprops={"color": "black", "linewidth": 4},
            boxprops={"edgecolor": "black"},
            # notch=True,
        )
    else:
        df = df.rename(columns={"Method": "x-axis", "misc.log_method": "Method"}, inplace=False)
        match plot_style:
            case PlotStyle.scatterplot:
                sns.set_palette("pastel")
                sns.scatterplot(
                    x="x-axis",
                    y=renamed_col_to_plot,
                    data=df,
                    ax=plot,
                    style="Method",
                    hue="Method",
                )
            case PlotStyle.lineplot:
                sns.set_palette("Set2")
                sns.lineplot(
                    x="x-axis",
                    y=renamed_col_to_plot,
                    data=df,
                    ax=plot,
                    style="Method",
                    hue="Method",
                )
            case PlotStyle.boxplot_hue:
                sns.boxplot(
                    x="x-axis",
                    y=renamed_col_to_plot,
                    data=df,
                    ax=plot,
                    whis=1.0,
                    hue="Method",
                    medianprops={"color": "black", "linewidth": 4},
                    boxprops={"edgecolor": "black"},
                )
    hatches = ["/", "\\", ".", "x", "/", "\\", ".", "x"]
    for hatch, patch in zip(hatches, plot.artists):
        # patch.set_hatch(hatch)
        patch.set_edgecolor("black")
        # patch.set_facecolor("lightgrey")
    for method, patch in zip(df["Method"].unique(), plot.artists):
        # Add dense 'x' hatching to the 'oracle' methods
        if isinstance(method, str) and "oracle" in method.lower():
            patch.set_hatch("xxx")

    # if you only want to set one ylim, then pass "nan" on the commandline for the other value
    plot.set_ylim(
        ymin=y_limits[0] if not math.isnan(y_limits[0]) else None,
        ymax=y_limits[1] if not math.isnan(y_limits[1]) else None,
    )
    plot.set_xlim(
        xmin=x_limits[0] if not math.isnan(x_limits[0]) else None,
        xmax=x_limits[1] if not math.isnan(x_limits[1]) else None,
    )
    if x_label is not None:
        plt.xlabel(x_label)
    if hide_left_ticks:
        plot.set_ylabel(None)
        plot.tick_params(left=False, labelleft=False)
    if plot_title is not None:
        plot.set_title(plot_title)
    if not with_legend and (legend := plot.get_legend()) is not None:
        legend.set_visible(False)

    return fig
