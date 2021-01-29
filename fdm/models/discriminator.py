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
        else:  # WGAN Loss is simply the difference between the means of the real and fake batches
            return real_scores.mean() - fake_scores.mean()

    def encoder_loss(self, fake: Tensor, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.criterion is DiscriminatorLoss.logistic:
            loss_real = F.softplus(real_scores)  # log(1-sigmoid(real_scores_out))
            loss_fake = -F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            return (loss_real + loss_fake).mean()
        elif self.criterion is DiscriminatorLoss.logistic_ns:
            loss_real = -F.softplus(-real_scores)  # log(sigmoid(real_scores_out))
            loss_fake = F.softplus(-fake_scores)  # -log(sigmoid(fake_scores_out))
            return (loss_real + loss_fake).mean()
        else:  # WGAN Loss is simply the difference between the means of the real and fake batches
            return fake_scores.mean() - real_scores.mean()
