from __future__ import annotations
from typing import Any

from fdm.models.base import ModelBase
from shared.configs.enums import DiscriminatorLoss
from torch import Tensor, nn
import torch.nn.functional as F

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
        scores_real = self.model(real)
        scores_fake = self.model(fake)
        scores_real = scores_real - scores_real.mean(dim=0)
        scores_fake = scores_real - scores_real.mean(dim=0)
        if self.criterion is DiscriminatorLoss.logistic:
            loss_fake = F.softplus(scores_real) # -log(1-sigmoid(fake_scores_out))
            loss_real = F.softplus(1 - scores_real) # -log(sigmoid(real_scores_out)) 
            return (loss_real + loss_fake).mean()
        else:  # WGAN Loss is simply the difference between the means of the real and fake batches
            return scores_real.mean() - scores_fake.mean()

    def generator_loss(self, fake: Tensor) -> Tensor:
        logits = self.model(fake)
        if self.criterion is DiscriminatorLoss.logistic:
            return F.softplus(-logits).mean()
        else:
            return logits.mean()
