from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision
import wandb
from torch import Tensor, nn

from fdm.configs import VaeArgs
from shared.utils import wandb_log

__all__ = ["log_images", "save_model", "restore_model", "weight_for_balance"]


def log_images(
    args: VaeArgs, image_batch, name, step, nsamples=64, nrows=8, monochrome=False, prefix=None
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
    args: VaeArgs, save_dir: Path, model: nn.Module, itr: int, sha: str, best: bool = False
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


def restore_model(args: VaeArgs, filename: Path, model: nn.Module):
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert args.enc_levels == args_chkpt["levels"]

    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]


def weight_for_balance(
    cluster_ids: Tensor, min_size: Optional[int] = None
) -> Tuple[Tensor, int, int, int]:
    unique, counts = torch.unique(cluster_ids, sorted=False, return_counts=True)
    n_clusters = int(unique.max() + 1)
    weights = torch.zeros((n_clusters,))
    # the higher the count the lower the weight to balance out
    weights[unique.long()] = 1 / counts.float()

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
