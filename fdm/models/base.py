import logging
from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, lr_scheduler

__all__ = ["ModelBase", "ModelBaseCosine", "EncodingSize", "SplitEncoding", "Reconstructions"]

log = logging.getLogger("MODELS")


class EncodingSize(NamedTuple):
    zs: int
    zy: int


class SplitEncoding(NamedTuple):
    zs: Tensor
    zy: Tensor


class Reconstructions(NamedTuple):
    all: Tensor
    rand_s: Tensor  # reconstruction with random s
    rand_y: Tensor  # reconstruction with random y
    zero_s: Tensor
    zero_y: Tensor
    just_s: Tensor


class ModelBase(nn.Module):

    default_kwargs = dict(optimizer_kwargs=dict(lr=1e-3, weight_decay=0))

    def __init__(self, model, optimizer_kwargs=None):
        super().__init__()
        self.model = model
        optimizer_kwargs = optimizer_kwargs or self.default_kwargs["optimizer_kwargs"]
        self.optimizer = Adam(self.model.parameters(), **optimizer_kwargs)

    def reset_parameters(self):
        def _reset_parameters(m: nn.Module):
            if hasattr(m.__class__, "reset_parameters") and callable(
                getattr(m.__class__, "reset_parameters")
            ):
                m.reset_parameters()

        self.model.apply(_reset_parameters)

    def step(self, grads=None):
        self.optimizer.step(grads)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forward(self, inputs):
        return self.model(inputs)


class ModelBaseCosine(ModelBase):
    def __init__(
        self, model, optimizer_kwargs=None, annealing_steps: int = 3_000, freeze: int = 3622
    ):
        super().__init__(model, optimizer_kwargs)
        self.annealing_steps = annealing_steps
        self.freeze = freeze
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.annealing_steps)
        # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 1000, 2)
        self.counter = 0

    def step(self, grads=None) -> float:
        super().step(grads)
        if self.counter == self.freeze:
            log.info("Freeze learning rate.")
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1_000_000)
        else:
            self.scheduler.step()
        self.counter += 1
        return float(self.scheduler.get_last_lr()[0])
