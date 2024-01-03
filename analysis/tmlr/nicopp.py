# %%
import sys

sys.path.append("..")

# %%
from pathlib import Path

from ranzen.wandb import RunsDownloader
from wandb_utils import (
    CustomMethod,
    Group,
    MethodName,
    Metrics,
    SpecialMetrics,
    download_groups,
    generate_table,
    plot,
    simple_concat,
)

# %%
suds_no_oracle = download_groups(
    RunsDownloader(project="support-matching", entity="predictive-analytics-lab"),
    {
        "our_method_no_balancing_2023-10-19": Group(
            MethodName.ours_no_balancing, "test/y_from_zy/", ""
        ),
        # "erm_suds_2023-10-06": Group(CustomMethod("ERM (suds)"), "test/pred_y/", ""),
        # "erm_suds_2023-10-09_zsdim1": Group(CustomMethod("ERM (suds)"), "test/y_from_zy/", ""),
    },
)
suds_no_oracle = suds_no_oracle.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

suds_with_oracle = download_groups(
    RunsDownloader(project="support-matching", entity="predictive-analytics-lab"),
    {
        "our_method_2023-10-02": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
        # "our_method_2023-10-02": Group(MethodName.ours_bag_oracle, "test/pred_y/", ""),
    },
)
suds_with_oracle = suds_with_oracle.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

# %%
hyaline_no_oracle = download_groups(
    RunsDownloader(project="hyaline", entity="predictive-analytics-lab"),
    {
        "erm_baseline_nicopp_2023-09-27": Group(MethodName.erm, "test/", ""),
        "dro_baseline_2023-09-27_eta_0.1": Group(MethodName.dro, "test/", ""),
        # "dro_baseline_2023-09-27_eta_0.2": Group(CustomMethod("DRO eta=0.2"), "test/", ""),
        # "dro_baseline_2023-09-27_eta_0.3": Group(CustomMethod("DRO eta=0.3"), "test/", ""),
        # "dro_baseline_2023-09-27_eta_0.4": Group(CustomMethod("DRO eta=0.4"), "test/", ""),
        # "dro_baseline_2023-09-27_eta_0.5": Group(CustomMethod("DRO eta=0.5"), "test/", ""),
    },
)
hyaline_with_oracle = download_groups(
    RunsDownloader(project="hyaline", entity="predictive-analytics-lab"),
    {
        "erm_superoracle_nicopp_2023-10-25_lower_lr": Group(MethodName.erm_oracle, "test/", ""),
        # "dfr_baseline_2023-10-17_l1_weight_0.001": Group(CustomMethod("DFR L1LossW=1.e-3"), "test/", ""),
        "dfr_baseline_2023-10-16_l1_weight_0.0001": Group(MethodName.dfr, "test/", ""),
        # "dfr_baseline_2023-10-18_l1_weight_1e-05": Group(CustomMethod("DFR L1LossW=1.e-5"), "test/", ""),
    },
)

# %%
data = simple_concat(suds_no_oracle, hyaline_no_oracle, suds_with_oracle, hyaline_with_oracle)

# %%
plot(
    data,
    metrics=[Metrics.acc],
    x_limits=(0.86, 0.89),
    file_format="pdf",
    fig_dim=(5.0, 1.9),
    file_prefix="nicopp",
    output_dir=Path("nicopp"),
    separator_after=2,
)

# %%
plot(
    data,
    metrics=[Metrics.rob_tpr_ovr],
    # x_limits=(0.48, 1),
    file_format="pdf",
    fig_dim=(5.0, 1.9),
    file_prefix="nicopp",
    output_dir=Path("nicopp"),
    separator_after=2,
)
# %%
generate_table(
    data,
    metrics=[
        SpecialMetrics.acc_table,
        SpecialMetrics.rob_tpr_ovr_table,
    ],
)

# %%
