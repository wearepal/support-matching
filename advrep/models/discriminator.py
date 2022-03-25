from __future__ import annotations
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from advrep.models.base import Model
from shared.configs.enums import DiscriminatorLoss
from shared.layers.aggregation import Aggregator

__all__ = ["Discriminator"]


class Discriminator(Model):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        aggregator: Optional[Aggregator],
        double_adv_loss: bool,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        criterion: DiscriminatorLoss = DiscriminatorLoss.logistic,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        model = backbone if aggregator is None else nn.Sequential(backbone, aggregator)
        super().__init__(
            model=model,
            lr=lr,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.backbone = backbone
        self.aggregator = aggregator

        self.double_adv_loss = double_adv_loss
        self.criterion = criterion

    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.criterion is DiscriminatorLoss.logistic:
            loss_real = -F.softplus(real_scores)  # -log(sigmoid(real_scores_out))
            loss_fake = F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            return loss_real.mean() + loss_fake.mean()
        elif self.criterion is DiscriminatorLoss.logistic_ns:
            loss_real = F.softplus(-real_scores)  # -log(sigmoid(real_scores_out))
            loss_fake = F.softplus(fake_scores)  # -log(1-sigmoid(fake_scores_out))
            return loss_real.mean() + loss_fake.mean()
        else:  # WGAN Loss is just the difference between the mean scores for the real and fake data
            return real_scores.mean() - fake_scores.mean()

    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        fake_scores = self.model(fake)
        real_scores: Tensor | None = None
        if self.double_adv_loss:
            real_scores = self.model(real)
        loss = fake.new_zeros(())
        if self.criterion is DiscriminatorLoss.logistic:
            loss -= F.softplus(fake_scores).mean()  # -log(1-sigmoid(fake_scores_out))
            if real_scores is not None:
                loss += F.softplus(real_scores).mean()  # log(1-sigmoid(real_scores_out))
        elif self.criterion is DiscriminatorLoss.logistic_ns:
            loss += F.softplus(-fake_scores).mean()  # -log(sigmoid(fake_scores_out))
            if real_scores is not None:
                loss -= F.softplus(-real_scores).mean()  # log(sigmoid(real_scores_out))
        else:  # WGAN Loss is just the difference between the scores for the fake and real data
            loss += fake_scores.mean()
            if real_scores is not None:
                loss -= real_scores.mean()
        return loss
