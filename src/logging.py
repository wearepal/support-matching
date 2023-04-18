from __future__ import annotations
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import TypeAlias

from conduit.data.datasets.vision.base import CdtVisionDataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as TF
import wandb
from loguru import logger

from src.data.data_module import DataModule

__all__ = [
    "WandbConf",
    "as_pretty_dict",
    "flatten_dict",
    "log_attention",
    "log_images",
    "reconstruct_cmd",
]
Run: TypeAlias = Union[
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


@dataclass
class WandbConf:
    name: Optional[str] = None
    mode: str = "online"
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = "support-matching"
    group: Optional[str] = None
    entity: Optional[str] = "predictive-analytics-lab"
    tags: Optional[List[str]] = None
    reinit: bool = True
    job_type: Optional[str] = None
    resume: Optional[str] = None
    dir: Optional[str] = "local_logging"
    notes: Optional[str] = None

    def init(
        self,
        raw_config: Optional[Dict[str, Any]] = None,
        keys_for_name: Tuple[str, ...] = (),
        suffix: Optional[str] = None,
    ) -> Run:
        if raw_config is not None and self.group is None:
            default_group = f"{raw_config['ds']['_target_'].lower()}_"
            if suffix is not None:
                default_group += suffix
            default_group += "_".join(
                raw_config[key]["_target_"].split(".")[-1].lower() for key in keys_for_name
            )
            logger.info(f"No wandb group set - using {default_group} as the inferred default.")
            self.group = default_group
        # TODO: not sure whether `reinit` really should be hardcoded
        return wandb.init(**asdict(self), config=raw_config, reinit=True)


def log_images(
    images: Tensor,
    *,
    dm: DataModule,
    name: str,
    step: int,
    nsamples: int | Sequence[int] = 64,
    ncols: int = 8,
    monochrome: bool = False,
    prefix: str | None = None,
    caption: str | None = None,
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


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _clean_up_dict(obj: Any) -> Any:
    """Convert enums to strings and filter out _target_."""
    if isinstance(obj, MutableMapping):
        return {key: _clean_up_dict(value) for key, value in obj.items() if key != "_target_"}
    elif isinstance(obj, Enum):
        return str(f"{obj.name}")
    elif OmegaConf.is_config(obj):  # hydra stores lists as omegaconf.ListConfig, so we convert here
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    return obj


def as_pretty_dict(data_class: object) -> dict:
    """Convert dataclass to a pretty dictionary."""
    return _clean_up_dict(asdict(data_class))


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program."""
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = internal_config.overrides.task
    return _join([program] + OmegaConf.to_container(args))  # type: ignore[operator]


def _join(split_command: List[str]) -> str:
    """Concatenate the tokens of the list split_command and return a string."""
    return " ".join(shlex.quote(arg) for arg in split_command)
