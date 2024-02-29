from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar
from typing_extensions import TypeAliasType

import torch.nn as nn

__all__ = ["BackboneFactory", "BackboneFactoryOut"]

M = TypeVar("M", bound=nn.Module, covariant=True)
BackboneFactoryOut = TypeAliasType("BackboneFactoryOut", tuple[M, int], type_params=(M,))


@dataclass
class BackboneFactory(ABC):
    @abstractmethod
    def __call__(self, input_dim: int) -> BackboneFactoryOut[nn.Module]:
        raise NotImplementedError()
