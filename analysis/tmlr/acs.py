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
        "ours_fifth_run_2024-03-07_no_balance": Group(
            MethodName.ours_no_balancing, "test/y_from_zy/", ""
        ),
    },
)
suds_no_oracle = suds_no_oracle.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

# %%
suds_with_oracle = download_groups(
    RunsDownloader(project="support-matching", entity="predictive-analytics-lab"),
    {
        "ours_fifth_run_2024-03-06": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
    },
)
suds_with_oracle = suds_with_oracle.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

# %%
hyaline_no_oracle = download_groups(
    RunsDownloader(project="hyaline", entity="predictive-analytics-lab"),
    {
        "baselines_erm_2024-03-06": Group(MethodName.erm, "test/", ""),
        # "baselines_jtt_2024_03-05": Group(MethodName.jtt, "test/", ""),
        "baselines_dro_0.05_2024-05-03": Group(MethodName.dro, "test/", ""),
        "baselines_gdro_2024-03-05": Group(MethodName.gdro, "test/", ""),
    },
)
# %%
hyaline_with_oracle = download_groups(
    RunsDownloader(project="hyaline", entity="predictive-analytics-lab"),
    {
        "baselines_dfr_2024-03-05": Group(MethodName.dfr, "test/", ""),
        "baselines_gdro_2024-03-06": Group(MethodName.gdro_oracle, "test/", ""),
    },
)
# %%
data = simple_concat(hyaline_no_oracle, suds_with_oracle, hyaline_with_oracle)

# %%
plot(
    data,
    metrics=[Metrics.acc],
    # x_limits=(0.86, 0.89),
    file_format="pdf",
    fig_dim=(5.0, 1.4),
    file_prefix="acs",
    output_dir=Path("acs"),
    separator_after=2,
)

# %%
plot(
    data,
    metrics=[Metrics.rob_tpr_ovr],
    # x_limits=(0.48, 1),
    file_format="pdf",
    fig_dim=(5.0, 1.5),
    file_prefix="acs",
    output_dir=Path("acs"),
    separator_after=2,
)

# %%
generate_table(
    data,
    metrics=[
        SpecialMetrics.acc_table,
        SpecialMetrics.rob_tpr_ovr_table,
    ],
    unicode=True,
)
# %%
