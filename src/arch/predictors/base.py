from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar
from typing_extensions import TypeAliasType

import torch.nn as nn

__all__ = ["PredictorFactory", "PredictorFactoryOut"]

M = TypeVar("M", bound=nn.Module, covariant=True)
PredictorFactoryOut = TypeAliasType("PredictorFactoryOut", tuple[M, int], type_params=(M,))


@dataclass(eq=False)
class PredictorFactory(ABC):
    @abstractmethod
    def __call__(
        self, input_dim: int, *, target_dim: int, batch_size: int
    ) -> PredictorFactoryOut[nn.Module]:
        raise NotImplementedError()
