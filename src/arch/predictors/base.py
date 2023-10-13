from abc import ABC, abstractmethod
from typing import TypeAlias, TypeVar

import torch.nn as nn

__all__ = ["PredictorFactory", "PredictorFactoryOut"]

M = TypeVar("M", bound=nn.Module, covariant=True)
PredictorFactoryOut: TypeAlias = tuple[M, int]


class PredictorFactory(ABC):
    @abstractmethod
    def __call__(
        self, input_dim: int, *, target_dim: int, batch_size: int
    ) -> PredictorFactoryOut[nn.Module]:
        raise NotImplementedError()
