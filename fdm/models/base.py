from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

__all__ = ["ModelBase", "EncodingSize", "SplitEncoding", "Reconstructions"]


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
