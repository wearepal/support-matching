from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, TypeVar
from typing_extensions import TypeAlias

import torch.nn as nn

__all__ = [
    "PredictorFactory",
    "PredictorFactoryOut",
]
M = TypeVar("M", bound=nn.Module)
PredictorFactoryOut: TypeAlias = Tuple[M, int]


@dataclass
class PredictorFactory:
    def __call__(
        self, input_dim: int, *, target_dim: int, **kwargs: Any
    ) -> Tuple[nn.Module, int]:  # PredictorFactoryOut:
        ...
