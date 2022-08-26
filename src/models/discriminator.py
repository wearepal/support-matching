from __future__ import annotations
from enum import Enum, auto
from typing import Any

from torch import Tensor, nn
import torch.nn.functional as F

from .base import Model

__all__ = [
    "Discriminator",
    "DiscriminatorLoss",
]


class DiscriminatorLoss(Enum):
    """Which type of adversarial loss to use."""

    WASSERSTEIN = auto()
    LOGISTIC_NS = auto()
    LOGISTIC_S = auto()


class Discriminator(Model):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        criterion: DiscriminatorLoss = DiscriminatorLoss.LOGISTIC_NS,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            lr=lr,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.criterion = criterion

    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.criterion is DiscriminatorLoss.LOGISTIC_S:
            loss_real = -F.softplus(real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        elif self.criterion is DiscriminatorLoss.LOGISTIC_NS:
            loss_real = F.softplus(-real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        else:  # WGAN Loss is just the difference between the mean scores for the real and fake data
            return real_scores.mean() - fake_scores.mean()

    def encoder_loss(self, fake: Tensor, *, real: Tensor | None) -> Tensor:
        fake_scores = self.model(fake)
        real_scores: Tensor | None = None
        if real is not None:
            real_scores = self.model(real)
        loss = fake.new_zeros(())
        if self.criterion is DiscriminatorLoss.LOGISTIC_S:
            loss -= F.softplus(fake_scores).mean()
            if real_scores is not None:
                loss += F.softplus(real_scores).mean()
        elif self.criterion is DiscriminatorLoss.LOGISTIC_NS:
            loss += F.softplus(-fake_scores).mean()
            if real_scores is not None:
                loss -= F.softplus(-real_scores).mean()  # log(sigmoid(real_scores_out))
        else:  # WGAN Loss is just the difference between the scores for the fake and real data
            loss += fake_scores.mean()
            if real_scores is not None:
                loss -= real_scores.mean()
        return loss
