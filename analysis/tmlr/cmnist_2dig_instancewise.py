# %%
from re import M
import sys

sys.path.append("..")

# %%
from math import nan

from ranzen.wandb import RunsDownloader
from wandb_utils import (
    Aggregation,
    CustomMethod,
    Group,
    MethodName,
    Metrics,
    PlotKwargs,
    concat_with_suffix,
    download_groups,
    plot,
    simple_concat,
)

# %%
wandb = RunsDownloader(project="suds", entity="predictive-analytics-lab")

# %%
data = download_groups(
    wandb,
    {
        "cmnist.SupportMatching.ranking-fdm.baseline_2021-09-22.subsampled": Group(MethodName.ours_with_bags),
        # "cmnist.SupportMatching.no-cluster-fdm.baseline_2021-09-22.subsampled": Group(MethodName.ours_no_balancing),
        # "cmnist.SupportMatching.perfect-cluster.baseline_2021-09-22.subsampled": Group(MethodName.ours_bag_oracle),
        "cmnist.SupportMatching.ranking-fdm.no_MIL_2021-09-22.subsampled": Group(MethodName.ours_no_bags),
        # "cmnist.SupportMatching.no-cluster-fdm.no_MIL_2021-09-22.subsampled": Group(MethodName.ours_no_balancing),
        # "cmnist.SupportMatching.perfect-cluster.no_MIL_2021-09-22.subsampled": Group(MethodName.ours_bag_oracle),
    },
)

# %%
plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (5, 0.7),
    "file_prefix": "cmnist_2v4_instancewise",
}

# %%
plot(data, metrics=[Metrics.acc], x_limits=(0.72, 1), **plot_kwargs)

# %%
plot(
    data,
    metrics=[Metrics.acc],
    x_limits=(nan, 1),
    agg=Aggregation.min,
    x_label="Robust Accuracy $\\rightarrow$",
    **plot_kwargs,
)

# %%
