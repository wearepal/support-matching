# %%
import sys

sys.path.append("..")

# %%
from math import nan
from pathlib import Path

from wandb_utils import Aggregation, Group, MethodName, Metrics, PlotKwargs, load_groups, plot

# %%
results_dir = Path("../../results/cmnist/3_digits/4miss/")
old_results_dir = Path("../../results/cmnist/3_digits/")

data = load_groups(
    {
        # results_dir / "ranking-fdm.mostly-tradidional.3dig_4miss.csv",
        # results_dir / "kmeans-fdm.mostly-tradidional.3dig_4miss.csv",
        results_dir / "no-cluster-fdm.mostly-tradidional.3dig_4miss.csv": Group(
            MethodName.ours_no_balancing
        ),
        results_dir / "perfect-cluster.mostly-tradidional.3dig_4miss.csv": Group(
            MethodName.ours_bag_oracle
        ),
        # old_results_dir / "cmnist_cnn_baseline_color_60epochs.csv",
        # old_results_dir / "cmnist_dro_baseline_color_60epochs.csv",
        results_dir / "cmnist_baseline_dro_color_eta_0.5_60epochs.csv": Group(
            MethodName.dro, metrics_suffix=" (dro)"
        ),
        results_dir / "cmnist_baseline_lff_color_60epochs.csv": Group(
            MethodName.lff, metrics_suffix=" (lff)"
        ),
        results_dir / "cmnist_baseline_cnn_color_60epochs.csv": Group(
            MethodName.erm, metrics_suffix=" (cnn)"
        ),
    }
)

# %%
plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (5, 1.25),
    "file_prefix": "cmnist_3dig_4miss",
}

# %%
plot(data, metrics=[Metrics.acc], x_limits=(nan, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.hgr], x_limits=(0, nan), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.prd], agg=Aggregation.max, x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.prr], agg=Aggregation.min, x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tprd], agg=Aggregation.max, x_limits=(-0.01, 1.01), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tprr], agg=Aggregation.min, x_limits=(-0.01, 1), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tnrd], agg=Aggregation.max, x_limits=(-0.01, 1.01), **plot_kwargs)

# %%
plot(data, metrics=[Metrics.tnrr], agg=Aggregation.min, x_limits=(-0.01, 1), **plot_kwargs)

# %%
