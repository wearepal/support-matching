from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence, TypeVar

from conduit.data.datasets.vision.base import CdtVisionDataset
import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms.functional as TF
import wandb

from shared.configs import Config
from shared.data.data_module import DataModule
from shared.utils import as_pretty_dict, flatten_dict

__all__ = [
    "log_attention",
    "log_images",
    "restore_model",
    "save_model",
]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


def log_images(
    cfg: Config,
    *,
    dm: DataModule,
    images: Tensor,
    name: str,
    step: int,
    nsamples: int | Sequence[int] = 64,
    ncols: int = 8,
    monochrome: bool = False,
    prefix: str | None = None,
    caption: str | None = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"

    if isinstance(dm.train, CdtVisionDataset):
        images = dm.denormalize(images)

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
    shw = [wandb.Image(TF.to_pil_image(i), caption=caption) for i in shw]
    wandb.log({prefix + name: shw}, step=step)


def log_attention(
    cfg: Config,
    *,
    dm: DataModule,
    images: Tensor,
    attention_weights: Tensor,
    name: str,
    step: int,
    nbags: int,
    border_width: int = 3,
    ncols: int = 8,
    prefix: str | None = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"

    if isinstance(dm.train, CdtVisionDataset):
        images = dm.denormalize(images)

    images = images.view(*attention_weights.shape, *images.shape[1:])
    images = images[:nbags].cpu()
    attention_weights = attention_weights[:nbags]
    padding = attention_weights.view(nbags, -1, 1, 1, 1)

    w_padding = padding.expand(-1, -1, 3, border_width, images.size(-1)).cpu()
    images = torch.cat([w_padding, images, w_padding], dim=-2)
    h_padding = padding.expand(-1, -1, 3, images.size(-2), border_width).cpu()
    images = torch.cat([h_padding, images, h_padding], dim=-1)

    shw = [
        torchvision.utils.make_grid(block, nrow=ncols, pad_value=1.0).clamp(0, 1)
        for block in images.unbind(dim=0)
    ]
    shw = [wandb.Image(TF.to_pil_image(image), caption=f"bag_{i}") for i, image in enumerate(shw)]
    wandb.log({prefix + name: shw}, step=step)


def save_model(cfg: Config, save_dir: Path, model: nn.Module, itr: int, best: bool = False) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{itr}.pth"
    save_dict = {
        "args": flatten_dict(as_pretty_dict(cfg)),
        "model": model.state_dict(),
        "itr": itr,
    }

    torch.save(save_dict, filename)

    return filename


M = TypeVar("M", bound=nn.Module)


def restore_model(cfg: Config, filename: Path, model: M) -> tuple[M, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert cfg.enc.levels == args_chkpt["enc.levels"]

    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]
