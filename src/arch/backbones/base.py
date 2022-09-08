from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar
from typing_extensions import TypeAlias

import torch.nn as nn

__all__ = [
    "BackboneFactory",
    "BackboneFactoryOut",
]

M = TypeVar("M", bound=nn.Module)
BackboneFactoryOut: TypeAlias = Tuple[M, int]


@dataclass
class BackboneFactory(Generic[M]):
    def __call__(self, input_dim: int) -> BackboneFactoryOut[M]:
        ...
