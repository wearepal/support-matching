"""Utility functions."""
from __future__ import annotations
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict
from enum import Enum
from functools import reduce
from math import gcd
from typing import Any, Iterable, TypeVar

from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from shared.configs.enums import ClusteringLabel
from shared.data.utils import labels_to_group_id

__all__ = [
    "AverageMeter",
    "as_pretty_dict",
    "count_parameters",
    "flatten_dict",
    "get_class_id",
    "get_data_dim",
    "get_joint_probability",
    "lcm",
    "prod",
    "readable_duration",
]

T = TypeVar("T")

Int = TypeVar("Int", Tensor, int)


def get_data_dim(data_loader: DataLoader) -> tuple[int, ...]:
    x = next(iter(data_loader))[0]
    x_dim = x.shape[1:]

    return tuple(x_dim)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prod(seq: Sequence[T]) -> T:
    if not seq:
        raise ValueError("seq cannot be empty")
    result = seq[0]
    for i in range(1, len(seq)):
        result *= seq[i]
    return result


def readable_duration(seconds: float, pad: str = "") -> str:
    """Produce human-readable duration."""
    if seconds < 10:
        return f"{seconds:.2g}s"
    seconds = int(round(seconds))

    parts = []

    time_minute = 60
    time_hour = 3600
    time_day = 86400
    time_week = 604800

    weeks, seconds = divmod(seconds, time_week)
    days, seconds = divmod(seconds, time_day)
    hours, seconds = divmod(seconds, time_hour)
    minutes, seconds = divmod(seconds, time_minute)

    if weeks:
        parts.append(f"{weeks}w")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not weeks and not days:
        parts.append(f"{minutes}m")
    if seconds and not weeks and not days and not hours:
        parts.append(f"{seconds}s")

    return pad.join(parts)


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


def lcm(denominators: Iterable[int]) -> int:
    """Least common multiplier."""
    return reduce(lambda a, b: a * b // gcd(a, b), denominators)


def get_class_id(*, s: Tensor, y: Tensor, s_count: int, to_cluster: ClusteringLabel) -> Tensor:
    if to_cluster == ClusteringLabel.s:
        class_id = s
    elif to_cluster == ClusteringLabel.y:
        class_id = y
    else:
        class_id = labels_to_group_id(s=s, y=y, s_count=s_count)
    return class_id.view(-1)


def get_joint_probability(*, s_probs: Tensor, y_probs: Tensor) -> Tensor:
    """Given probabilities for s and y, return the joint probability.

    This function has been tested to be compatible with get_class_id() above.
    """
    # take the outer product of s_probs and y_probs
    return (s_probs.unsqueeze(-2) * y_probs.unsqueeze(-1)).flatten(start_dim=-2)
