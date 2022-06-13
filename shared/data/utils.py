from typing import TypeVar
from typing_extensions import Literal

from torch import Tensor

__all__ = [
    "labels_to_group_id",
    "group_id_to_label",
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
