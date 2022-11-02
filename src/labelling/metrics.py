from __future__ import annotations

from conduit.models.utils import prefix_keys
from loguru import logger
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import wandb

__all__ = ["compute_accuracy", "evaluate"]


def count_cooccurrances(
    test_group_ids: npt.NDArray[np.int32], *, clusters: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Count how often every possible pair of group ID and cluster ID co-occur."""
    counts: dict[tuple[int, int], int] = {}
    max_group = 0
    max_cluster = 0
    for group in np.unique(test_group_ids):
        for cluster in np.unique(clusters):
            counts[(group, cluster)] = np.count_nonzero(
                (test_group_ids == group) & (clusters == cluster)
            )
            if cluster > max_cluster:
                max_cluster = cluster
        if group > max_group:
            max_group = group
    counts_np = np.zeros((max_group + 1, max_cluster + 1), dtype=np.int32)
    for (group, cluster), count in counts.items():
        counts_np[group, cluster] = count
    return counts_np


def compute_accuracy(
    test_group_ids: npt.NDArray[np.int32], *, clusters: npt.NDArray[np.int32]
) -> float:
    # in order to solve the assignment problem, we find the assignment that maximizes counts
    counts = count_cooccurrances(test_group_ids, clusters=clusters)
    row_ind, col_ind = linear_sum_assignment(counts, maximize=True)
    num_corectly_assigned = counts[row_ind, col_ind].sum()
    return num_corectly_assigned / len(test_group_ids)


def evaluate(
    y_true: npt.NDArray[np.int32],
    *,
    y_pred: npt.NDArray[np.int32],
    use_wandb: bool = True,
    prefix: str | None = None,
) -> None:
    metrics = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Accuracy": compute_accuracy(y_true, clusters=y_pred),
    }
    if prefix is not None:
        metrics = prefix_keys(metrics, prefix=prefix, sep="/")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3g}")
    logger.info("---")
    if use_wandb:
        wandb.log(metrics)
    return metrics
