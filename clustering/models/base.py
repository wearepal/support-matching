import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

__all__ = ["ModelBase", "Encoder"]

log = logging.getLogger("MODELS")


class ModelBase(nn.Module):

    default_kwargs = dict(optimizer_kwargs=dict(lr=1e-3, weight_decay=0))
    optimizer: Adam

    def __init__(
        self, model: nn.Module, optimizer_kwargs: Optional[Dict[str, float]] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs or self.default_kwargs["optimizer_kwargs"]
        self._reset_optimizer(self.optimizer_kwargs)

    def _reset_optimizer(self, optimizer_kwargs: Dict[str, float]) -> None:
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_kwargs
        )

    def reset_parameters(self) -> None:
        def _reset_parameters(m: nn.Module) -> None:
            if hasattr(m.__class__, "reset_parameters") and callable(
                getattr(m.__class__, "reset_parameters")
            ):
                m.reset_parameters()

        self.model.apply(_reset_parameters)

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.optimizer.step(grads)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)

    def freeze_initial_layers(
        self, num_layers: int, optimizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        assert isinstance(self.model, nn.Sequential), "model isn't indexable"
        log.info(f"Freezing {num_layers} out of {len(self.model)} layers.")
        for block in self.model[:num_layers]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self._reset_optimizer(optimizer_kwargs or self.optimizer_kwargs)


class Encoder(nn.Module):
    @abstractmethod
    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        """Encode the given input."""

    @abstractmethod
    def fit(
        self, train_data: DataLoader, epochs: int, device: torch.device, use_wandb: bool
    ) -> None:
        """Train the encoder on the given data."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out gradients."""

    @abstractmethod
    def step(self, grads: Optional[Tensor] = None) -> None:
        """Do a step with the optimizer."""

    @abstractmethod
    def freeze_initial_layers(self, num_layers: int, optimizer_kwargs: Dict[str, Any]) -> None:
        """Freeze the initial layers of the model."""

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)
