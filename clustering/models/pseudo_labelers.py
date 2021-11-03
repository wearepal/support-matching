"""Labelers"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from shared.utils import dot_product

__all__ = ["PseudoLabeler", "RankingStatistics", "CosineSimThreshold"]


class PseudoLabeler(nn.Module):
    """Base class for labelers."""

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Get pseudo label for all combinations of the samples in the batch z."""

    def get_label(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Get pseudo label for z1 and z2."""
        return self.__call__(z)


class RankingStatistics(PseudoLabeler):
    def __init__(self, k_num: int):
        super().__init__()
        self.k_num = k_num

    def _get_topk(self, z: Tensor) -> Tensor:
        return torch.sort(torch.topk(torch.flatten(z, 1), k=self.k_num).indices).values

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        topk_z = self._get_topk(z.abs())
        labels = (topk_z == topk_z[:, None, :]).all(dim=-1).float()
        return labels, torch.ones_like(labels)


class CosineSimThreshold(PseudoLabeler):
    def __init__(self, upper_threshold: float, lower_threshold: float):
        super().__init__()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z = z.flatten(start_dim=1)
        cosine_sim = dot_product(z, z[:, None, :])
        over = (cosine_sim > self.upper_threshold).float()
        under = (cosine_sim < self.lower_threshold).float()
        mask = over + under
        labels = over  # this will ensure that the label for all samples in `under` is 0
        return labels, mask
