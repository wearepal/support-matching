from collections.abc import Iterator
from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.types import Number

__all__ = ["cat", "soft_prediction", "to_item", "to_numpy", "full_class_path"]


DT = TypeVar("DT", bound=np.number | np.bool_)


@overload
def to_numpy(tensor: Tensor, *, dtype: DT) -> npt.NDArray[DT]: ...


@overload
def to_numpy(tensor: Tensor, *, dtype: None = ...) -> npt.NDArray: ...


def to_numpy(tensor: Tensor, *, dtype: DT | None = None) -> npt.NDArray[DT] | npt.NDArray:
    arr = tensor.detach().cpu().numpy()
    if dtype is not None:
        arr.astype(dtype)
    return arr


def to_item(tensor: Tensor) -> Number:
    return tensor.detach().cpu().item()


def cat(
    *ls: list[Tensor], dim: int = 0, device: torch.device | str | None = None
) -> Iterator[Tensor]:
    for ls_ in ls:
        yield torch.cat(ls_, dim=dim).to(device=device)


def soft_prediction(logits: Tensor) -> Tensor:
    logits = logits.squeeze(1) if logits.ndim == 2 else torch.atleast_1d(logits)
    return logits.sigmoid() if logits.ndim == 1 else logits.softmax(dim=1)


def full_class_path(obj: object) -> str:
    class_ = obj.__class__
    return f"{class_.__module__}.{class_.__qualname__}"
