from typing import TypeVar
from typing_extensions import Protocol, runtime_checkable

import torch.nn as nn

__all__ = ["ModelFactory"]

M_co = TypeVar("M_co", bound=nn.Module, covariant=True)


@runtime_checkable
class ModelFactory(Protocol[M_co]):
    def __call__(self, input_dim: int, *, target_dim: int) -> M_co:
        ...
