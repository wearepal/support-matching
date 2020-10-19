from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from fdm.configs import FdmArgs
from shared.utils import class_id_to_label

__all__ = [
    "get_all_num_samples",
    "log_images",
    "restore_model",
    "save_model",
    "weight_for_balance",
    "weights_with_counts",
]


def save_model(
    args: FdmArgs, save_dir: Path, model: nn.Module, itr: int, sha: str, best: bool = False
) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{itr}.pth"
    save_dict = {
        "args": args.as_dict(),
        "sha": sha,
        "model": model.state_dict(),
        "itr": itr,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(args: FdmArgs, filename: Path, model: nn.Module):
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert args.enc_levels == args_chkpt["levels"]

    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]


def _get_weights(cluster_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    unique, counts = torch.unique(cluster_ids, sorted=False, return_counts=True)
    n_clusters = int(unique.max() + 1)
    weights = torch.zeros((n_clusters,))
    # the higher the count the lower the weight to balance out
    weights[unique.long()] = 1 / counts.float()
    return weights, counts, unique


def weight_for_balance(
    cluster_ids: Tensor, min_size: Optional[int] = None
) -> Tuple[Tensor, int, int, int]:
    weights, counts, unique = _get_weights(cluster_ids)

    n_used_clusters = counts.size(0)
    if min_size is not None:
        smallest_used_cluster = int(counts.max())
        for cluster, count in zip(unique, counts):
            count_int = int(count)
            if count_int < min_size:
                print(f"Dropping cluster {cluster} with only {count_int} elements.")
                print("Consider setting --oversample to True (or improve clustering).")
                weights[cluster] = 0  # skip this cluster
                n_used_clusters -= 1
            elif count_int < smallest_used_cluster:
                smallest_used_cluster = count_int
    else:
        smallest_used_cluster = int(counts.min())
    return weights[cluster_ids.long()], n_used_clusters, smallest_used_cluster, int(counts.max())


def weights_with_counts(cluster_ids: Tensor) -> Tuple[Tensor, Dict[int, Tuple[float, int]]]:
    weights, counts, unique = _get_weights(cluster_ids)
    w_and_c = {int(i): (float(weights[i]), int(c)) for i, c in zip(unique, counts)}
    return weights[cluster_ids.long()], w_and_c


def get_all_num_samples(
    quad_w_and_c: Dict[int, Tuple[float, int]],
    y_w_and_c: Dict[int, Tuple[float, int]],
    s_count: int,
) -> List[int]:
    # multiply the quad weights with the correct y weights
    combined_w_and_c = []
    for class_id, (weight, count) in quad_w_and_c.items():
        y_weight, _ = y_w_and_c[class_id_to_label(class_id, s_count, "y")]
        combined_w_and_c.append((weight * y_weight, count))

    # compute what the intended proportions were for a balanced batch
    intended_proportions = [weight * count for weight, count in combined_w_and_c]
    sum_int_prop = sum(intended_proportions)
    intended_proportions = [prop / sum_int_prop for prop in intended_proportions]
    # compute what the size of the dataset would be if we were to use all the samples from the
    # individual clusters and the correct proportions for all the other clusters
    return [round(count / prop) for (_, count), prop in zip(combined_w_and_c, intended_proportions)]
