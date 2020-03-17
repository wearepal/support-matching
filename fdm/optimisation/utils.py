from pathlib import Path
from typing import Tuple

import torch
import torchvision
from torch import nn

import wandb
from fdm.configs import VaeArgs
from fdm.utils import wandb_log

__all__ = ["get_data_dim", "log_images", "save_model", "restore_model"]


def get_data_dim(data_loader) -> Tuple[int, ...]:
    x = next(iter(data_loader))[0]
    x_dim = x.shape[1:]

    return tuple(x_dim)


def log_images(
    args: VaeArgs, image_batch, name, step, nsamples=64, nrows=8, monochrome=False, prefix=None
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]

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
    args: VaeArgs, save_dir: Path, vae: nn.Module, epoch: int, sha: str, best: bool = False,
) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{epoch}.pth"
    save_dict = {
        "args": args.as_dict(),
        "sha": sha,
        "vae": vae.state_dict(),
        "epoch": epoch,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(args: VaeArgs, filename: Path, vae: nn.Module):
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert args.levels == args_chkpt["levels"]
    assert args.level_depth == args_chkpt["level_depth"]

    vae.load_state_dict(chkpt["vae"])
    return vae, chkpt["epoch"]
