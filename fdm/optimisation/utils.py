from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torchvision
import wandb

from shared.configs import Config, FdmDataset, ReconstructionLoss
from shared.utils import StratifiedSampler, as_pretty_dict, flatten_dict, wandb_log

__all__ = [
    "get_stratified_sampler",
    "log_images",
    "restore_model",
    "save_model",
]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


def log_images(
    cfg: Config,
    images,
    name,
    step,
    nsamples: int | Sequence[int] = 64,
    ncols: int = 8,
    monochrome: bool = False,
    prefix: str | None = None,
    caption: str | None = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"

    if cfg.enc.recon_loss == ReconstructionLoss.ce:
        images = images.argmax(dim=1).float() / 255
    else:
        if cfg.data.dataset in (FdmDataset.celeba, FdmDataset.isic):
            images = 0.5 * images + 0.5

    if monochrome:
        images = images.mean(dim=1, keepdim=True)

    if isinstance(nsamples, int):
        blocks = [images[:nsamples]]
    else:
        blocks = []
        start_index = 0
        for num in nsamples:
            blocks.append(images[start_index : start_index + num])
            start_index += num

    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = [
        torchvision.utils.make_grid(block, nrow=ncols, pad_value=1.0).clamp(0, 1).cpu()
        for block in blocks
    ]
    shw = [
        wandb.Image(torchvision.transforms.functional.to_pil_image(i), caption=caption) for i in shw
    ]
    wandb_log(cfg.misc, {prefix + name: shw}, step=step)


def save_model(
    cfg: Config, save_dir: Path, model: nn.Module, itr: int, sha: str, best: bool = False
) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{itr}.pth"
    save_dict = {
        "args": flatten_dict(as_pretty_dict(cfg)),
        "sha": sha,
        "model": model.state_dict(),
        "itr": itr,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(cfg: Config, filename: Path, model: nn.Module) -> tuple[nn.Module, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert cfg.enc.levels == args_chkpt["enc.levels"]

    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]


def get_stratified_sampler(
    group_ids: Tensor, oversample: bool, batch_size: int, min_size: int | None = None
) -> StratifiedSampler:
    unique, counts = torch.unique(group_ids, sorted=False, return_counts=True)

    n_used_clusters = counts.size(0)
    multipliers = {}
    if min_size is not None:
        smallest_used_cluster = int(counts.max())
        for cluster, count in zip(unique, counts):
            count_int = int(count)
            if count_int < min_size:
                LOGGER.info(f"Dropping cluster {cluster} with only {count_int} elements.")
                LOGGER.info("Consider setting --oversample to True (or improve clustering).")
                # set this cluster's multiplier to 0 so that it is skipped
                multipliers[int(cluster)] = 0
                n_used_clusters -= 1
            elif count_int < smallest_used_cluster:
                smallest_used_cluster = count_int
    else:
        smallest_used_cluster = int(counts.min())
    group_size = int(counts.max()) if oversample else smallest_used_cluster
    num_samples = n_used_clusters * group_size
    assert num_samples > batch_size, f"not enough training samples ({num_samples}) to fill a batch"
    return StratifiedSampler(
        group_ids.squeeze().tolist(), group_size, replacement=oversample, multipliers=multipliers
    )
