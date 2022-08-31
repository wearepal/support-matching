from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Protocol

from ranzen import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor
import torch.nn.functional as F

from src.mmd import MMDKernel, mmd2

from .base import Model

__all__ = [
    "GanLoss",
    "MmdDiscriminator",
    "NeuralDiscriminator",
]


class Discriminator(Protocol):
    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        ...

    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        ...


@dataclass(eq=False)
class MmdDiscriminator(Discriminator, DcModule):
    mmd_kernel: MMDKernel = MMDKernel.rq
    mmd_scales: List[float] = field(default_factory=list)
    mmd_wts: List[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    @implements(Discriminator)
    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        return torch.zeros((), device=fake.device)

    @implements(Discriminator)
    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        return mmd2(
            x=fake,
            y=real,
            kernel=self.mmd_kernel,
            scales=self.mmd_scales,
            wts=self.mmd_wts,
            add_dot=self.mmd_add_dot,
        )


class GanLoss(Enum):
    """Which type of adversarial loss to use."""

    WASSERSTEIN = auto()
    LOGISTIC_NS = auto()
    LOGISTIC_S = auto()


@dataclass(eq=False)
class NeuralDiscriminator(Discriminator, Model):
    criterion: GanLoss = GanLoss.LOGISTIC_NS

    @implements(Discriminator)
    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.criterion is GanLoss.LOGISTIC_S:
            loss_real = -F.softplus(real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        elif self.criterion is GanLoss.LOGISTIC_NS:
            loss_real = F.softplus(-real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        return real_scores.mean() - fake_scores.mean()

    @implements(Discriminator)
    def encoder_loss(self, fake: Tensor, *, real: Tensor | None) -> Tensor:
        fake_scores = self.model(fake)
        real_scores: Tensor | None = None
        if real is not None:
            real_scores = self.model(real)
        loss = fake.new_zeros(())
        if self.criterion is GanLoss.LOGISTIC_S:
            loss -= F.softplus(fake_scores).mean()
            if real_scores is not None:
                loss += F.softplus(real_scores).mean()
        elif self.criterion is GanLoss.LOGISTIC_NS:
            loss += F.softplus(-fake_scores).mean()
            if real_scores is not None:
                loss -= F.softplus(-real_scores).mean()
        else:
            loss += fake_scores.mean()
            if real_scores is not None:
                loss -= real_scores.mean()
        return loss