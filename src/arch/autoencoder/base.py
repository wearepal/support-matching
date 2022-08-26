from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, TypeVar
from typing_extensions import TypeAlias

import torch.nn as nn

E = TypeVar("E", bound=nn.Module)
D = TypeVar("D", bound=nn.Module)
AeFactoryOut: TypeAlias = Tuple[E, D, int]


@dataclass
class AeFactory:
    def __call__(
        self,
        input_shape: tuple[int, int, int],
    ) -> AeFactoryOut:
        ...
