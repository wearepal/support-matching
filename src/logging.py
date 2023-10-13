from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, TypeAlias, Union

from conduit.data.datasets.vision.base import CdtVisionDataset
from ranzen.hydra import reconstruct_cmd
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as TF
import wandb

from src.data.data_module import DataModule

__all__ = ["WandbConf", "log_attention", "log_images"]

Run: TypeAlias = Union[
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


WandbMode = Enum("WandbMode", ["online", "offline", "disabled"])


@dataclass
class WandbConf:
    name: Optional[str] = None
    mode: WandbMode = WandbMode.online
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = "support-matching"
    group: Optional[str] = None
    entity: Optional[str] = "predictive-analytics-lab"
    tags: Optional[list[str]] = None
    reinit: bool = True
    job_type: Optional[str] = None
    resume: Optional[str] = None
    dir: Optional[str] = "local_logging"
    notes: Optional[str] = None

    def init(
        self,
        raw_config: Optional[dict[str, Any]] = None,
        cfgs_for_group: tuple[object, ...] = (),
        with_tag: Optional[str] = None,
    ) -> Run:
        if self.tags is None:
            self.tags = []
        self.tags.extend(cfg_obj.__class__.__name__ for cfg_obj in cfgs_for_group)
        if with_tag is not None:
            self.tags.append(with_tag)
        # TODO: not sure whether `reinit` really should be hardcoded
        self.reinit = True
        kwargs = asdict(self)
        kwargs["mode"] = self.mode.name
        if raw_config is not None:
            raw_config["cmd"] = reconstruct_cmd()
        return wandb.init(**kwargs, config=raw_config)


def log_images(
    images: Tensor,
    *,
    dm: DataModule,
    name: str,
    step: int,
    nsamples: Union[int, Sequence[int]] = 64,
    ncols: int = 8,
    monochrome: bool = False,
    prefix: Optional[str] = None,
    caption: Optional[str] = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train/" if prefix is None else f"{prefix}/"

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
    dm: DataModule,
    *,
    images: Tensor,
    attention_weights: Tensor,
    name: str,
    step: int,
    nbags: int,
    border_width: int = 3,
    ncols: int = 8,
    prefix: Optional[str] = None,
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
