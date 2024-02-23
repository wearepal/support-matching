# %%
import sys

sys.path.append("..")

# %%
from pathlib import Path

from ranzen.wandb import RunsDownloader
from wandb_utils import CustomMethod, Group, MethodName, Metrics, download_groups, plot

# %%
downloader = RunsDownloader(project="support-matching", entity="predictive-analytics-lab")

# %%
data = download_groups(
    downloader,
    {
        # "our_method_2023-10-02": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
        "our_method_no_balancing_2023-10-19": Group(
            MethodName.ours_no_balancing, "test/y_from_zy/", ""
        ),
        "ours_2023-12-19_no_recon_loss": Group(
            CustomMethod("Without recon loss"), "test/y_from_zy/", ""
        ),
        # "ours_2023-12-19_no_zs": Group(CustomMethod("no zs split"), "test/y_from_zy/", ""),
        "ours_2023-12-19_no_y_predictor": Group(
            CustomMethod("Without $y$-predictor"), "test/y_from_zy/", ""
        ),
        "ours_2024-01-05_no_disc_loss": Group(
            CustomMethod("Without disc loss"), "test/y_from_zy/", ""
        ),
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
    file_prefix="nicopp",
    output_dir=Path("nicopp") / "more_ablation",
)

# %%
