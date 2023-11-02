# %%
import sys

sys.path.append("..")

# %%
from math import nan
from pathlib import Path

from ranzen.wandb import RunsDownloader
from wandb_utils import (
    Aggregation,
    CustomMethod,
    Group,
    MethodName,
    Metrics,
    PlotKwargs,
    download_groups,
    load_groups,
    plot,
    simple_concat,
)

# %%
results_dir = Path("../../results/cmnist/2v4/subsampled-0.66cont/")

data = load_groups(
    {
        # results_dir / "kmeans-fdm.simplified.strong_subs.subsampled.csv",
        results_dir / "no-cluster-fdm.simplified.strong_subs.subsampled.csv": Group(
            MethodName.ours_no_balancing
        ),
        results_dir / "ranking-fdm.simplified.strong_subs.subsampled.csv": Group(
            MethodName.ours_clustering
        ),
        results_dir / "perfect-cluster.simplified.strong_subs.subsampled.csv": Group(
            MethodName.ours_bag_oracle
        ),
        results_dir / "cmnist_baseline_cnn_color_60epochs.csv": Group(
            MethodName.erm, metrics_suffix=" (cnn)"
        ),
        #     results_dir / "cmnist_baseline_dro_color_eta_0.1_60epochs.csv",
        results_dir / "cmnist_baseline_gdro_color_60epochs.csv": Group(
            MethodName.gdro, metrics_suffix=" (gdro)"
        ),
        results_dir / "cmnist_baseline_lff_color_60epochs.csv": Group(
            MethodName.lff, metrics_suffix=" (lff)"
        ),
    }
)

# %%
wandb = RunsDownloader(project="fdm-hydra", entity="predictive-analytics-lab")
data_ = download_groups(
    wandb,
    {
        "ranking-fdm-6.overcluster6.subsampled": Group(CustomMethod("Ours (Clustering; k=6)")),
        # "kmeans-fdm-6.overcluster6.subsampled",
        "ranking-fdm.overcluster8.subsampled": Group(CustomMethod("Ours (Clustering; k=8)")),
        # "kmeans-fdm-8.overcluster8.subsampled",
    },
)
data = simple_concat(data, data_)
# wandb = RunsDownloader(project="suds", entity="predictive-analytics-lab")
# data_ = download_groups(
#     wandb,
#     {"cmnist.SupportMatching.new_ae_arch.subsampled": Group(CustomMethod("new arch??"))},
# )
# data = simple_concat(data, data_)

# %%
plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (5, 2.5),
    "file_prefix": "cmnist_2v4_partial_overcluster",
}

# %%
plot(data, metrics=[Metrics.acc], x_limits=(nan, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.acc], agg=Aggregation.min, x_label="Robust Accuracy $\\rightarrow$", x_limits=(nan, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tprr], x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tnrr], x_limits=(0.83, 1), **plot_kwargs)

# %%
