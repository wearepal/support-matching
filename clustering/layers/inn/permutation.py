from typing import Optional, Tuple

import torch
from torch import Tensor

from shared.utils import is_positive_int

from .bijector import Bijector

__all__ = ["RandomPermutation", "ReversePermutation"]


class Permutation(Bijector):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation: Tensor, dim: int = 1) -> None:
        if permutation.dim() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self) -> Tensor:
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs: Tensor, permutation: Tensor, dim: int) -> Tensor:
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(dim, len(permutation))
            )
        batch_size = inputs.shape[0]
        return torch.index_select(inputs, dim, permutation)

    def _forward(
        self, inputs: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y = self._permute(inputs, self._permutation, self._dim)

        return y, sum_ldj

    def _inverse(
        self, inputs: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y = self._permute(inputs, self._inverse_permutation, self._dim)

        return y, sum_ldj


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, in_channels: int, dim: int = 1) -> None:
        if not is_positive_int(in_channels):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(in_channels), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, in_channels: int, dim: int = 1) -> None:
        if not is_positive_int(in_channels):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(in_channels - 1, -1, -1), dim)
