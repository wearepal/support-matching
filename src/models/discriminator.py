from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from typing_extensions import override

import ot
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.mmd import MMDKernel, mmd2

from .base import Model, OptimizerCfg

__all__ = [
    "BinaryDiscriminator",
    "DiscOptimizerCfg",
    "GanLoss",
    "MmdDiscriminator",
    "NeuralDiscriminator",
    "WassersteinDiscriminator",
]


class DiscType(Enum):
    MMD = auto()
    """Maximum mean discrepancy."""
    EMD = auto()
    """Earth mover's distance."""
    NEURAL = auto()
    """Adversarial neural network."""


class BinaryDiscriminator(ABC):
    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        return torch.zeros((), device=fake.device)

    @abstractmethod
    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        raise NotImplementedError()


@dataclass(repr=False, eq=False)
class MmdDiscriminator(BinaryDiscriminator):
    mmd_kernel: MMDKernel = MMDKernel.rq
    mmd_scales: list[float] = field(default_factory=list)
    mmd_wts: list[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    @override
    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        return mmd2(
            x=fake,
            y=real,
            kernel=self.mmd_kernel,
            scales=self.mmd_scales,
            wts=self.mmd_wts,
            add_dot=self.mmd_add_dot,
        )


@dataclass(repr=False, eq=False)
class WassersteinDiscriminator(BinaryDiscriminator):
    @override
    def encoder_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        size_batch = fake.shape[0]
        weights = torch.ones(size_batch, device=fake.device) / size_batch
        metric_cost_matrix: Tensor = ot.dist(fake, real)

        return ot.emd2(weights, weights, metric_cost_matrix)  # type: ignore


class GanLoss(Enum):
    """Which type of adversarial loss to use."""

    WASSERSTEIN = auto()
    LOGISTIC_NS = auto()
    LOGISTIC_S = auto()
    LS = auto()


def _maybe_spectral_norm(module: nn.Module, *, name: str = "weight"):
    if hasattr(module, name):
        torch.nn.utils.parametrizations.spectral_norm(module, name=name)


@dataclass
class DiscOptimizerCfg(OptimizerCfg):
    """These are the parameters to `NeuralDiscriminator` which are configurable by hydra."""

    criterion: GanLoss = GanLoss.LOGISTIC_NS


@dataclass(repr=False, eq=False)
class NeuralDiscriminator(BinaryDiscriminator, Model):
    opt: DiscOptimizerCfg  # overriding the definition in `Model`

    def __post_init__(self) -> None:
        if self.opt.criterion is GanLoss.WASSERSTEIN:
            self.model.apply(_maybe_spectral_norm)
        super().__post_init__()

    @override
    def discriminator_loss(self, fake: Tensor, *, real: Tensor) -> Tensor:
        real_scores = self.model(real)
        fake_scores = self.model(fake)
        if self.opt.criterion is GanLoss.LOGISTIC_S:
            loss_real = -F.softplus(real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        elif self.opt.criterion is GanLoss.LOGISTIC_NS:
            loss_real = F.softplus(-real_scores)
            loss_fake = F.softplus(fake_scores)
            return loss_real.mean() + loss_fake.mean()
        elif self.opt.criterion is GanLoss.LS:
            return 0.5 * ((real_scores - 1).pow(2).mean() + (fake_scores).pow(2).mean())
        return real_scores.mean() - fake_scores.mean()

    @override
    def encoder_loss(self, fake: Tensor, *, real: Optional[Tensor]) -> Tensor:
        fake_scores = self.model(fake)
        real_scores: Optional[Tensor] = None
        if real is not None:
            real_scores = self.model(real)
        loss = fake.new_zeros(())
        if self.opt.criterion is GanLoss.LOGISTIC_S:
            loss -= F.softplus(fake_scores).mean()
            if real_scores is not None:
                loss += F.softplus(real_scores).mean()
        elif self.opt.criterion is GanLoss.LOGISTIC_NS:
            loss += F.softplus(-fake_scores).mean()
            if real_scores is not None:
                loss -= F.softplus(-real_scores).mean()
        elif self.opt.criterion is GanLoss.LS:
            loss += 0.5 * (fake_scores - 1).square().mean()
            if real_scores is not None:
                loss += 0.5 * real_scores.square().mean()
        else:
            loss += fake_scores.mean()
            if real_scores is not None:
                loss -= real_scores.mean()
        return loss
