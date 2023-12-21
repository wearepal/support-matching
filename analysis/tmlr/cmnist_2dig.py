# %%
import sys

sys.path.append("..")

# %%
from math import nan
from pathlib import Path

from wandb_utils import (
    Group,
    MethodName,
    Metrics,
    PlotKwargs,
    SpecialMetrics,
    generate_table,
    load_groups,
    plot,
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
generate_table(
    data,
    metrics=[
        SpecialMetrics.acc_table,
        SpecialMetrics.rob_acc_table,
        Metrics.prr,
        Metrics.tprr,
        Metrics.tnrr,
    ],
)

# %%
plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (5.0, 1.6),
    "file_prefix": "cmnist_2v4_partial",
}

# %%
plot(data, metrics=[Metrics.acc], x_limits=(nan, 1), **plot_kwargs)

# %%
plot(data, metrics=[SpecialMetrics.rob_acc], x_limits=(nan, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.prr], x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tprr], x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tnrr], x_limits=(nan, 1), **plot_kwargs)

# %%
