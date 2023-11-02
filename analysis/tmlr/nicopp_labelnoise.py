# %%
import sys

sys.path.append("..")

# %%
from ranzen.wandb import RunsDownloader
from wandb_utils import Group, MethodName, Metrics, PlotStyle, download_groups, plot, simple_concat

# %%
downloader = RunsDownloader(project="support-matching", entity="predictive-analytics-lab")

# %%
baseline = download_groups(
    downloader,
    {
        "our_method_2023-10-02": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
    },
)
baseline["labeller/UniformLabelNoiser.level"] = 0.0
with_noise = download_groups(
    downloader,
    {
        "our_method_2023-10-02_noise_0.2": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
        "our_method_2023-10-02_noise_0.4": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
        # "our_method_2023-10-02_noise_0.6": Group(MethodName.ours_bag_oracle, "test/y_from_zy/", ""),
        "our_method_2023-10-30_noise_0.6_seeded_noise": Group(
            MethodName.ours_bag_oracle, "test/y_from_zy/", ""
        ),
    },
)
data = simple_concat(baseline, with_noise)
data = data.rename(columns={"Robust_OvR_TPR": "Robust OvR TPR"})

# %%
plot(
    data,
    groupby="labeller/UniformLabelNoiser.level",
    metrics=[Metrics.acc, Metrics.rob_tpr_ovr],
    # metrics=[Metrics.acc],
    x_label="$\\rho$",
    # x_limits=(0.48, 1),
    plot_style=PlotStyle.boxplot_hue,
    file_format="pdf",
    fig_dim=(3.0, 5.0),
    file_prefix="nicopp_labelnoise",
    with_legend=False,
)

# %%
