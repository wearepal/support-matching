from __future__ import annotations
from typing import Any, Dict, Iterator, List, Tuple, TypeVar, Union
from typing_extensions import Literal

from conduit.data.datasets.utils import infer_sample_cls
from conduit.data.structures import NamedSample
import torch
from torch import Tensor

__all__ = [
    "group_id_to_label",
    "labels_to_group_id",
    "resolve_device",
    "sample_converter",
    "to_device",
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


def to_device(
    *args: Tensor,
    device: str | torch.device | int,
) -> Iterator[Tensor]:
    device = resolve_device(device)
    for arg in args:
        yield arg.to(device, non_blocking=True)


def sample_converter(sample: Union[Any, Tuple[Any, ...], List[Any], Dict[str, Any]]) -> NamedSample:
    sample_cls = infer_sample_cls(sample)
    if isinstance(sample, (tuple, list)):
        sample_d = dict(zip(["y", "s"], sample[1:]))
        return sample_cls(x=sample[0], **sample_d)
    return sample_cls(sample)
