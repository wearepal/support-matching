# %%
import sys

sys.path.append("..")
# %%
from math import nan
from pathlib import Path

from ranzen.wandb import RunsDownloader
from wandb_utils import (
    Group,
    MethodName,
    Metrics,
    PlotKwargs,
    SpecialMetrics,
    download_groups,
    generate_table,
    plot,
)

# %%
wandb = RunsDownloader(project="suds", entity="predictive-analytics-lab")
# The sequence of the methods to be presented: Our method (no balancing), Our method (clustering), gDRO (clustering) aka GEORGE, ERM,
# Our method (Batch Oracle), gDRO (Oracle). Ideally, you can shade the areas for Oracles

# Our Method (No Balancing)
data = download_groups(
    wandb,
    group_mapping={
        "celeba.SupportMatching.balanced-False.new_ae_arch_more_iters.no_unsmiling_males": Group(
            name=MethodName.ours_no_balancing
        ),
        "celeba.SupportMatching.balanced-with-clustering.new_ae_arch_more_iters_hierarchical_clustering.no_unsmiling_males": Group(
            name=MethodName.ours_clustering
        ),
        "celeba.erm.context_mode_unlabelled.erm_no_context_no_reg.no_unsmiling_males": Group(
            name=MethodName.erm, metrics_suffix=" (erm)"
        ),
        "celeba.gdro.context_mode_cluster_labels.cluster_and_gdro.real_gdro.no_unsmiling_males": Group(
            name=MethodName.george, metrics_suffix=" (gdro)"
        ),
        "celeba.gdro.context_mode_ContextMode.unlabelled..gdro_tests.no_unsmiling_males": Group(
            name=MethodName.gdro, metrics_suffix=" (gdro)"
        ),
        "celeba.SupportMatching.balanced-True.new_ae_arch_more_iters.no_unsmiling_males": Group(
            name=MethodName.ours_bag_oracle
        ),
        "celeba.gdro.context_mode_ground_truth.oracle_gdro.celeba_gdro.no_unsmiling_males": Group(
            name=MethodName.gdro_oracle, metrics_suffix=" (gdro)"
        ),
    },
)

# %%

plot_kwargs: PlotKwargs = {
    "file_format": "pdf",
    "fig_dim": (4, 2),
    "file_prefix": "celeba_gender_smiling",
    "sens_attr": "Male",
    "separator_after": 4,
}
plot_title = "Missing source: unsmiling males"
directory = "no_unsmiling_males"

# %%
plot(
    data.copy(),
    metrics=[SpecialMetrics.rob_acc],
    x_limits=(0.65, 1.0),
    **plot_kwargs,
    output_dir=Path("celeba") / directory,
)

# %%
plot(
    data,
    metrics=[SpecialMetrics.rob_acc],
    x_limits=(nan, 1.0),
    **plot_kwargs,
    output_dir=Path("celeba") / "supmat" / directory,
    fillna=True,
)

# %%
plot(
    data,
    metrics=[Metrics.acc, Metrics.prr, Metrics.tprr, Metrics.tnrr],
    x_limits=(nan, 1.0),
    **plot_kwargs,
    output_dir=Path("celeba") / "supmat" / directory,
    fillna=True,
)

# %%
generate_table(
    data,
    metrics=[
        SpecialMetrics.acc_table,
        SpecialMetrics.rob_acc_table,
        SpecialMetrics.prr_table,
        SpecialMetrics.tprr_table,
        SpecialMetrics.tnrr_table,
    ],
    sens_attr="Male",
)

# %%
