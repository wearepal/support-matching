from pathlib import Path
from typing import Tuple, Union, Dict

from lapjv import lapjv  # pylint: disable=no-name-in-module
import numpy as np
import torch
import torchvision
from torch import Tensor
from typing_extensions import Literal
import wandb

from shared.utils import wandb_log, save_results, ClusterResults
from clustering.models import Model
from clustering.configs import ClusterArgs

__all__ = [
    "convert_and_save_results",
    "count_occurances",
    "find_assignment",
    "get_class_id",
    "get_cluster_label_path",
    "log_images",
    "restore_model",
    "save_model",
]


def log_images(
    args: ClusterArgs, image_batch, name, step, nsamples=64, nrows=8, monochrome=False, prefix=None
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]

    if args.recon_loss == "ce":
        images = images.argmax(dim=1).float() / 255
    else:
        if args.dataset in ("celeba", "ssrp", "genfaces"):
            images = 0.5 * images + 0.5

    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    wandb_log(
        args,
        {prefix + name: [wandb.Image(torchvision.transforms.functional.to_pil_image(shw))]},
        step=step,
    )


def save_model(
    args: ClusterArgs, save_dir: Path, model: Model, epoch: int, sha: str, best: bool = False
) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{epoch}.pth"
    save_dict = {
        "args": args.as_dict(),
        "sha": sha,
        "model": model.state_dict(),
        "epoch": epoch,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(args: ClusterArgs, filename: Path, model: Model) -> Tuple[Model, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert args.enc_levels == args_chkpt["enc_levels"]
    model.load_state_dict(chkpt["model"])
    return model, chkpt["epoch"]


def count_occurances(
    counts: np.ndarray,
    preds: np.ndarray,
    s: Tensor,
    y: Tensor,
    s_count: int,
    to_cluster: Literal["s", "y", "both"],
) -> Tuple[np.ndarray, Tensor]:
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
) -> Tuple[float, "np.ndarray[np.int64]", Dict[str, Union[float, str]]]:
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


def get_class_id(
    *, s: Tensor, y: Tensor, s_count: int, to_cluster: Literal["s", "y", "both"]
) -> Tensor:
    if to_cluster == "s":
        class_id = s
    elif to_cluster == "y":
        class_id = y
    else:
        class_id = y * s_count + s
    return class_id.view(-1)


def get_cluster_label_path(args: ClusterArgs, save_dir: Path) -> Path:
    if args.cluster_label_file:
        return Path(args.cluster_label_file)
    else:
        return save_dir / "cluster_results.pth"


def convert_and_save_results(
    args: ClusterArgs,
    cluster_label_path: Path,
    results: Tuple[Tensor, Tensor, Tensor],
    enc_path: Path,
    context_acc: float,
    test_acc: float = float("nan"),
) -> Path:
    clusters, s, y = results
    s_count = args._s_dim if args._s_dim > 1 else 2
    class_ids = get_class_id(s=s, y=y, s_count=s_count, to_cluster=args.cluster)
    cluster_results = ClusterResults(
        flags=args.as_dict(),
        cluster_ids=clusters,
        class_ids=class_ids,
        enc_path=enc_path,
        context_acc=context_acc,
        test_acc=test_acc,
    )
    return save_results(save_path=cluster_label_path, cluster_results=cluster_results)
