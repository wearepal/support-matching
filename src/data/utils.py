from __future__ import annotations
from typing import TypeVar
from typing_extensions import Literal

import torch
from torch import Tensor

__all__ = [
    "labels_to_group_id",
    "group_id_to_label",
    "resolve_device",
]

I = TypeVar("I", Tensor, int)


def labels_to_group_id(*, s: I, y: I, s_count: int) -> I:
    assert s_count > 1
    return y * s_count + s


def group_id_to_label(group_id: I, *, s_count: int, label: Literal["s", "y"]) -> I:
    assert s_count > 1
    if label == "s":
        return group_id % s_count
    else:
        return group_id // s_count


def resolve_device(device: str | torch.device | int) -> torch.device:
    if isinstance(device, int):
        use_gpu = torch.cuda.is_available() and device >= 0
        device = torch.device(device if use_gpu else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return device
