from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal

from conduit.data.datasets.utils import infer_sample_cls
from conduit.data.structures import NamedSample
import torch
from torch import Tensor

__all__ = [
    "EvalTuple",
    "LabelPair",
    "group_id_to_label",
    "labels_to_group_id",
    "resolve_device",
    "sample_converter",
    "to_device",
]


def labels_to_group_id(*, s: I, y: I, s_count: int) -> I:
    assert s_count > 1
    return y * s_count + s


P = TypeVar("P", Tensor, None)


@dataclass(eq=False, init=False)
class EvalTuple(Generic[P]):
    def __init__(self, y_true: Tensor, *, y_pred: Tensor, s: Tensor, probs: P = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.s = s
        self.probs = probs


I = TypeVar("I", Tensor, int)


@dataclass(eq=False)
class LabelPair(Generic[I]):
    s: I
    y: I

    def __iter__(self) -> Iterator[I]:
        yield from (self.s, self.y)


@overload
def group_id_to_label(group_id: I, *, s_count: int, label: Literal["s"]) -> I:
    ...


@overload
def group_id_to_label(group_id: I, *, s_count: int, label: Literal["y"]) -> I:
    ...


@overload
def group_id_to_label(group_id: I, *, s_count: int, label: Literal[None] = ...) -> LabelPair[I]:
    ...


def group_id_to_label(
    group_id: I, *, s_count: int, label: Optional[Literal["s", "y"]] = None
) -> Union[I, LabelPair[I]]:
    assert s_count > 1
    if label is None:
        y = group_id_to_label(group_id=group_id, s_count=s_count, label="y")
        s = group_id_to_label(group_id=group_id, s_count=s_count, label="s")
        return LabelPair(s=s, y=y)
    elif label == "s":
        return group_id % s_count
    if isinstance(group_id, Tensor):
        return group_id.div(s_count, rounding_mode="floor")
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
