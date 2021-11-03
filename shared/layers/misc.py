from __future__ import annotations

from typing import Sequence, Tuple

import torch.nn as nn
from torch import Tensor
from typing_extensions import Literal

__all__ = ["View", "UnitNormLayer"]


class View(nn.Module):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)


class UnitNormLayer(nn.Module):
    def __init__(self, p: int | float | Literal["fro", "nuc"], dim: int | Sequence[int] = 1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x / x.norm(dim=self.dim, keepdim=True, p=self.p)
