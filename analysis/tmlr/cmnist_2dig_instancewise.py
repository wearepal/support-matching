# %%
import sys

sys.path.append("..")

# %%
from math import nan
from pathlib import Path

from ranzen.wandb import RunsDownloader
from wandb_utils import Aggregation, Group, MethodName, Metrics, PlotKwargs, download_groups, plot

# %%
wandb = RunsDownloader(project="suds", entity="predictive-analytics-lab")

# %%
data = download_groups(
    wandb,
    {
        "cmnist.SupportMatching.ranking-fdm.baseline_2021-09-22.subsampled": Group(
            MethodName.ours_with_bags
        ),
        # "cmnist.SupportMatching.no-cluster-fdm.baseline_2021-09-22.subsampled": Group(MethodName.ours_no_balancing),
        # "cmnist.SupportMatching.perfect-cluster.baseline_2021-09-22.subsampled": Group(MethodName.ours_bag_oracle),
        "cmnist.SupportMatching.ranking-fdm.no_MIL_2021-09-22.subsampled": Group(
            MethodName.ours_no_bags
        ),
        # "cmnist.SupportMatching.no-cluster-fdm.no_MIL_2021-09-22.subsampled": Group(MethodName.ours_no_balancing),
        # "cmnist.SupportMatching.perfect-cluster.no_MIL_2021-09-22.subsampled": Group(MethodName.ours_bag_oracle),
    },
)

# %%
plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (5, 0.7),
    "file_prefix": "cmnist_2v4_instancewise",
    "output_dir": Path("cmnist") / "subgroup_bias_nomil",
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
