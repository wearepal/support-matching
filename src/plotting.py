import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

__all__ = ["plot_histogram", "plot_histogram_by_source"]


def plot_histogram(
    vector: torch.Tensor,
    step: int,
    prefix: str = "train",
    cols: int = 3,
    rows: int = 6,
    bins: int = 30,
) -> None:
    """Plot a histogram over the batch"""
    vector = torch.flatten(vector, start_dim=1).detach().cpu()
    vector_np = vector.numpy()
    matplotlib.use("Agg")
    fig, plots = plt.subplots(figsize=(8, 12), ncols=cols, nrows=rows)
    for j in range(rows):
        for i in range(cols):
            _ = plots[j][i].hist(vector_np[:, j * cols + i], bins=np.linspace(-15, 15, bins))
    fig.tight_layout()

    log_dict = {
        f"{prefix}_histogram": fig,
        f"{prefix}_xi_min": vector_np.min(),
        f"{prefix}_xi_max": vector_np.max(),
        f"{prefix}_xi_nans": float(bool(np.isnan(vector_np).any())),
        f"{prefix}_xi_tensor": vector,
    }
    wandb.log(log_dict, step=step)


def _plot_histo(soft_preds: torch.Tensor, n_bins: int, pred_target: int | None = None) -> Figure:
    class_dim = soft_preds.size(1) if soft_preds.ndim > 1 else 1
    if class_dim > 1:
        if pred_target is not None:
            probs = soft_preds[:, pred_target]
        else:
            probs = soft_preds.max(dim=1).values
    else:
        if pred_target is not None:
            probs = soft_preds if pred_target == 1 else (1 - soft_preds)
        else:
            probs = torch.where(soft_preds > 0.5, soft_preds, 1 - soft_preds)
        probs = probs.view((probs.size(0),))
    probs_np = probs.detach().cpu().numpy()
    fig, plot = plt.subplots(dpi=200, figsize=(6, 4))
    plot.hist(probs_np, bins=n_bins, range=(0, 1))
    plot.set_xlim(left=0, right=1)
    fig.tight_layout()
    return fig


def plot_histogram_by_source(
    soft_preds: torch.Tensor,
    *,
    s: torch.Tensor,
    y: torch.Tensor,
    step: int,
    name: str,
    n_bins: int = 15,
) -> None:
    """Plot a histogram over the batch"""
    logging_dict = {}
    for s_value in s.unique():
        for y_value in y.unique():
            preds_for_source = soft_preds[(y == y_value) & (s == s_value)]
            fig = _plot_histo(preds_for_source, n_bins=n_bins)
            logging_dict[f"{name}confidence_s={s_value}_y={y_value}"] = wandb.Image(fig)
            fig = _plot_histo(preds_for_source, n_bins=n_bins, pred_target=y_value)
            logging_dict[f"{name}prob_true_class_s={s_value}_y={y_value}"] = wandb.Image(fig)

        preds_for_subgroup = soft_preds[(s == s_value)]
        fig = _plot_histo(preds_for_subgroup, n_bins=n_bins)
        logging_dict[f"{name}confidence_s={s_value}"] = wandb.Image(fig)
    wandb.log(logging_dict, step=step)
