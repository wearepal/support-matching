"""Utility functions."""
import collections
import os
import random
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader
from typing_extensions import Literal, Protocol

from shared.configs import Misc

__all__ = [
    "AverageMeter",
    "ModelFn",
    "RunningAverageMeter",
    "class_id_to_label",
    "count_parameters",
    "flatten",
    "get_data_dim",
    "inf_generator",
    "label_to_class_id",
    "prod",
    "random_seed",
    "readable_duration",
    "save_checkpoint",
    "wandb_log",
]

T = TypeVar("T")

Int = TypeVar("Int", Tensor, int)


class ModelFn(Protocol):
    def __call__(self, input_dim: int, target_dim: int) -> nn.Module:
        ...


def get_data_dim(data_loader: DataLoader) -> Tuple[int, ...]:
    x = next(iter(data_loader))[0]
    x_dim = x.shape[1:]

    return tuple(x_dim)


def label_to_class_id(*, s: Int, y: Int, s_count: int) -> Int:
    assert s_count > 1
    return y * s_count + s


def class_id_to_label(class_id: Int, s_count: int, label: Literal["s", "y"]) -> Int:
    assert s_count > 1
    if label == "s":
        return class_id % s_count
    else:
        return class_id // s_count


def wandb_log(
    args: Union[bool, Misc], row: Dict[str, Any], step: int, commit: Optional[bool] = None
) -> None:
    """Wrapper around wandb's log function"""
    if isinstance(args, bool):
        if not args:  # this has to be nested in order to make sure `args` is not bool in `elif`
            return
    elif not args.use_wandb:
        return
    wandb.log(row, commit=commit, step=step)


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


class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable: Iterable[T]) -> Iterator[T]:
    """Get DataLoaders in a single infinite loop.

    for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iter(iterable)
    # try to take one element to ensure that the iterator is not empty
    first_value = next(iterator, None)
    if first_value is not None:
        yield first_value
    else:
        raise RuntimeError("The given iterable is empty.")
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


def count_parameters(model):
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_seed(seed_value, use_cuda) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


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


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
