from __future__ import annotations
from pathlib import Path

from lapjv import lapjv
import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from torch import Tensor

from shared.configs import ClusteringLabel, MiscConfig
from shared.utils import class_id_to_label, label_to_class_id

__all__ = [
    "cluster_metrics",
    "count_occurances",
    "find_assignment",
    "get_class_id",
    "get_cluster_label_path",
]


def count_occurances(
    counts: np.ndarray,
    preds: np.ndarray,
    s: Tensor,
    y: Tensor,
    s_count: int,
    to_cluster: ClusteringLabel,
) -> tuple[np.ndarray, Tensor]:
    """Count how often cluster IDs coincide with the class IDs.

    All possible combinations are accounted for.
    """
    class_id = get_class_id(s=s, y=y, s_count=s_count, to_cluster=to_cluster)
    indices, batch_counts = np.unique(
        np.stack([class_id.numpy().astype(np.int64), preds]), axis=1, return_counts=True
    )
    counts[tuple(indices)] += batch_counts
    return counts, class_id


def find_assignment(
    counts: np.ndarray, num_total: int
) -> tuple[float, "np.ndarray[np.int64]", dict[str, float | str]]:
    """Find an assignment of cluster to class such that the overall accuracy is maximized."""
    # row_ind maps from class ID to cluster ID: cluster_id = row_ind[class_id]
    # col_ind maps from cluster ID to class ID: class_id = row_ind[cluster_id]
    row_ind, col_ind, result = lapjv(-counts)
    best_acc = -result[0] / num_total
    assignment = (f"{class_id}->{cluster_id}" for class_id, cluster_id in enumerate(row_ind))
    logging_dict = {
        "Best acc": best_acc,
        "class ID -> cluster ID": ", ".join(assignment),
    }
    return best_acc, col_ind, logging_dict


def get_class_id(*, s: Tensor, y: Tensor, s_count: int, to_cluster: ClusteringLabel) -> Tensor:
    if to_cluster == ClusteringLabel.s:
        class_id = s
    elif to_cluster == ClusteringLabel.y:
        class_id = y
    else:
        class_id = label_to_class_id(s=s, y=y, s_count=s_count)
    return class_id.view(-1)


def get_cluster_label_path(misc: MiscConfig, save_dir: Path) -> Path:
    if misc.cluster_label_file:
        return Path(misc.cluster_label_file)
    else:
        return save_dir / "cluster_results.pth"


def cluster_metrics(
    *,
    cluster_ids: np.ndarray,
    counts: np.ndarray,
    true_class_ids: np.ndarray,
    num_total: int,
    s_count: int,
    to_cluster: ClusteringLabel,
) -> tuple[float, dict[str, float], dict[str, str | float]]:
    # find best assignment for cluster to classes
    best_acc, best_ass, logging_dict = find_assignment(counts, num_total)
    metrics = {"Accuracy": best_acc}
    pred_class_ids = best_ass[cluster_ids]  # use the best assignment to get the class IDs

    conf_mat = confusion_matrix(true_class_ids, pred_class_ids, normalize="all")
    logging_dict["confusion matrix"] = f"\n{conf_mat}\n"

    nmi = normalized_mutual_info_score(labels_true=true_class_ids, labels_pred=pred_class_ids)
    metrics["NMI"] = nmi
    ari = adjusted_rand_score(labels_true=true_class_ids, labels_pred=pred_class_ids)
    metrics["ARI"] = ari
    acc_per_class = confusion_matrix(true_class_ids, pred_class_ids, normalize="true").diagonal()
    assert acc_per_class.ndim == 1
    if to_cluster == ClusteringLabel.both:
        for class_id_, acc in enumerate(acc_per_class):
            y_ = class_id_to_label(class_id_, s_count=s_count, label="y")
            s_ = class_id_to_label(class_id_, s_count=s_count, label="s")
            metrics[f"Acc y={y_} s={s_}"] = acc

    logging_dict.update(metrics)
    return best_acc, metrics, logging_dict
