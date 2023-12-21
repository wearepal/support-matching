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
        "celeba.SupportMatching.balanced-False.new_ae_arch_more_iters.no_smiling_females": Group(
            name=MethodName.ours_no_balancing
        ),
        "celeba.SupportMatching.balanced-with-clustering.new_ae_arch_more_iters_hierarchical_clustering.no_smiling_females": Group(
            name=MethodName.ours_clustering
        ),
        "celeba.SupportMatching.balanced-True.new_ae_arch_more_iters.no_smiling_females": Group(
            name=MethodName.ours_bag_oracle
        ),
        "celeba.erm.context_mode_unlabelled.erm_no_context_no_reg.no_smiling_females": Group(
            name=MethodName.erm, metrics_suffix=" (erm)"
        ),
        "celeba.gdro.context_mode_cluster_labels.cluster_and_gdro.real_gdro.no_smiling_females": Group(
            name=MethodName.george, metrics_suffix=" (gdro)"
        ),
        "celeba.gdro.context_mode=ContextMode.unlabelled..gdro_tests.no_smiling_females": Group(
            name=MethodName.gdro, metrics_suffix=" (gdro)"
        ),
        "celeba.gdro.context_mode_ground_truth.oracle_gdro.celeba_gdro.no_smiling_females": Group(
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
    "output_dir": "no_smiling_females",
}
plot_title = "Missing source: smiling females"

# %%
plot(
    data,
    metrics=[SpecialMetrics.rob_acc],
    x_limits=(0.65, 1.0),
    **{**plot_kwargs, "output_dir": Path("cutoff") / plot_kwargs["output_dir"]},
)

# %%
plot(
    data,
    metrics=[Metrics.acc, SpecialMetrics.rob_acc, Metrics.prr, Metrics.tprr, Metrics.tnrr],
    x_limits=(nan, 1.0),
    **plot_kwargs,
    fillna=True,
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
    sens_attr="Male",
)

# %%
