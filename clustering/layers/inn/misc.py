from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .bijector import Bijector

__all__ = ["Flatten", "ConstantAffine"]


class Flatten(Bijector):
    """Flatten the input (except batch dimension)."""

    first_input: bool

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("orig_shape", torch.tensor([-1, 0, 0, 0], dtype=torch.int64))
        self.first_input = True

    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.first_input:
            orig_shape = torch.tensor(x.shape, dtype=torch.int64)
            self.orig_shape[1:] = orig_shape[1:]
            self.first_input = False
        y = x.flatten(start_dim=1)
        return y, sum_ldj

    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = y.view(self.orig_shape[0], self.orig_shape[1], self.orig_shape[2], self.orig_shape[3])
        return x, sum_ldj


class ConstantAffine(Bijector):
    def __init__(self, scale: Tensor, shift: Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))
        self.register_buffer("shift", torch.as_tensor(shift))

    def logdetjac(self) -> Tensor:
        return self.scale.log().flatten().sum()

    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y = self.scale * x + self.shift

        if sum_ldj is not None:
            sum_ldj -= self.logdetjac()
        return y, sum_ldj

    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = (y - self.shift) / self.scale

        if sum_ldj is not None:
            sum_ldj += self.logdetjac()
        return x, sum_ldj
