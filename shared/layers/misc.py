from typing import Tuple

from torch import Tensor
import torch.nn as nn

__all__ = ["View"]


class View(nn.Module):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)
