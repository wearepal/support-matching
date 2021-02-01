from __future__ import annotations
from typing import Any, cast

from torch import Tensor, nn
import torch.nn.functional as F

from fdm.models.base import ModelBase
from shared.configs.enums import DiscriminatorLoss

__all__ = ["Discriminator"]


class Discriminator(ModelBase):
    def __init__(
        self,
        model: nn.Module,
        criterion: DiscriminatorLoss = DiscriminatorLoss.logistic,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model, optimizer_kwargs=optimizer_kwargs)
        self.criterion = criterion

    def discriminator_loss(self, fake: Tensor, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.criterion is DiscriminatorLoss.logistic:
            loss_real = -F.softplus(real_scores)  # -log(sigmoid(real_scores_out))
            loss_fake = F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            return (loss_real + loss_fake).mean()
        elif self.criterion is DiscriminatorLoss.logistic_ns:
            loss_real = F.softplus(-real_scores)  # -log(sigmoid(real_scores_out))
            loss_fake = F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            return (loss_real + loss_fake).mean()
        else:  # WGAN Loss is just the difference between the mean scores for the real and fake data
            return real_scores.mean() - fake_scores.mean()

    def encoder_loss(self, fake: Tensor, real: Tensor, use_real: bool) -> Tensor:
        fake_scores = self.model(fake)
        real_scores: Tensor | None = None
        if use_real:
            real_scores = self.model(real)
        loss = fake_scores.new_zeros(())
        if self.criterion is DiscriminatorLoss.logistic:
            loss -= F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            if real_scores is not None:
                loss += F.softplus(real_scores)  # log(1-sigmoid(real_scores_out))
        elif self.criterion is DiscriminatorLoss.logistic_ns:
            loss += F.softplus(-fake_scores)  # -log(sigmoid(fake_scores_out))
            if real_scores is not None:
                loss -= F.softplus(-real_scores)  # log(sigmoid(real_scores_out))
        else:  # WGAN Loss is just the difference between the scores for the fake and real data
            loss += fake_scores
            if real_scores is not None:
                loss -= real_scores
        return loss.mean()
