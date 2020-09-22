from typing import Tuple

import torch.nn as nn
from torch import Tensor

__all__ = ["View"]


class View(nn.Module):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)
