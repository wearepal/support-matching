# %%
import sys

sys.path.append("..")

# %%
from ranzen.wandb import RunsDownloader
from wandb_utils import Group, MethodName, Metrics, download_groups, plot

# %%
downloader = RunsDownloader(project="support-matching", entity="predictive-analytics-lab")

# %%
data = download_groups(
    downloader,
    {
        "our_method_2023-10-02": Group(MethodName.ours_with_bags, "test/y_from_zy/", ""),
        "our_method_no_bags_2023-10-06": Group(MethodName.ours_no_bags, "test/y_from_zy/", ""),
    },
)
data = data.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

# %%
plot(
    data,
    metrics=[Metrics.acc, Metrics.rob_tpr_ovr],
    # metrics=[Metrics.acc],
    # x_label="noise level",
    # x_limits=(0.48, 1),
    # plot_style=PlotStyle.boxplot_hue,
    file_format="pdf",
    fig_dim=(5.0, 1.0),
    file_prefix="nicopp_bag_ablation",
)

# %%
