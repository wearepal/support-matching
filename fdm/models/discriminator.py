from __future__ import annotations
from typing import Any

from torch import Tensor, nn
import torch.nn.functional as F

from fdm.models.base import ModelBase
from shared.configs.enums import DiscriminatorLoss

__all__ = ["Discriminator"]


class Discriminator(ModelBase):
    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: dict[str, Any] | None,
        criterion: DiscriminatorLoss,
    ):
        super().__init__(model, optimizer_kwargs=optimizer_kwargs)
        self.criterion = criterion

    def _apply_criterion(self, logits_fake: Tensor, logits_real: Tensor) -> Tensor:
        if self.criterion is DiscriminatorLoss.minimax:
            ones = logits_fake.new_ones((logits_fake.size(0), 1))
            zeros = logits_fake.new_zeros((logits_fake.size(0), 1))
            loss_fake = F.binary_cross_entropy_with_logits(logits_fake, zeros, reduction="mean")
            loss_real = F.binary_cross_entropy_with_logits(logits_real, ones, reduction="mean")
            return loss_real + loss_fake
        else:  # WGAN Loss is simply the difference between the means of the real and fake batches
            return logits_real.mean() - logits_fake.mean()

    def routine(self, fake: Tensor, real: Tensor) -> Tensor:
        logits_real = self.model(real)
        logits_fake = self.model(fake)
        return self._apply_criterion(logits_real=logits_real, logits_fake=logits_fake)

    def predict(self, fake: Tensor) -> Tensor:
        logits = self.model(fake)
        if self.criterion is DiscriminatorLoss.minimax:
            zeros = fake.new_zeros((logits.size(0), 1))
            return -F.binary_cross_entropy_with_logits(logits, zeros, reduction="mean")
        else:
            return logits.mean()
