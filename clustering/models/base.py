from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Any, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.utils.optimizers import RAdam

__all__ = ["ModelBase", "Encoder"]


class ModelBase(nn.Module):

    default_kwargs = dict(optimizer_kwargs=dict(lr=1e-3, weight_decay=0))
    optimizer: RAdam

    def __init__(self, model: nn.Module, optimizer_kwargs=None):
        super().__init__()
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs or self.defaul_kwargs["optimizer_kwargs"]
        self._reset_optimizer(self.optimizer_kwargs)

    def _reset_optimizer(self, optimizer_kwargs) -> None:
        self.optimizer = RAdam(
            filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_kwargs,
        )

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

    def freeze_initial_layers(self, num_layers: int, optimizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        assert isinstance(self.model, nn.Sequential), "model isn't indexable"
        print(f"Freezing {num_layers} out of {len(self.model)} layers.")
        for block in self.model[:num_layers]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self._reset_optimizer(optimizer_kwargs or self.optimizer_kwargs)


class Encoder(nn.Module):
    @abstractmethod
    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        """Encode the given input."""

    @abstractmethod
    def fit(self, train_data: DataLoader, epochs: int, device: torch.device, use_wandb: bool):
        """Train the encoder on the given data."""

    @abstractmethod
    def zero_grad(self):
        """Zero out gradients."""

    @abstractmethod
    def step(self, grads=None):
        """Do a step with the optimizer."""

    @abstractmethod
    def freeze_initial_layers(self, num_layers: int, optimizer_kwargs: Dict[str, Any]) -> None:
        """Freeze the initial layers of the model."""

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)
