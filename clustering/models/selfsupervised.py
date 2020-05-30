from typing import Any, Dict
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .base import Encoder
from .classifier import Classifier


class SelfSupervised(Encoder):
    """Encoder trained with self-supervision."""
    def __init__(self, classifier: Classifier):
        super().__init__()
        encoder = classifier.model
        self.fc = encoder.fc
        encoder.fc = nn.Identity()
        classifier.model = encoder
        self.classifier = classifier

    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        """Encode the given input."""
        return self.classifier.model(x)

    def fit(self, train_data: DataLoader, epochs: int, device: torch.device, use_wandb: bool):
        """Train the encoder on the given data."""
        self.classifier.model.fc = self.fc
        self.classifier.fit(train_data, epochs, device, use_wandb)
        self.fc = self.classifier.model.fc
        self.classifier.model.fc = nn.Identity()

    def zero_grad(self):
        """Zero out gradients."""
        self.classifier.zero_grad()

    def step(self, grads=None):
        """Do a step with the optimizer."""
        self.classifier.step(grads=grads)

    def freeze_initial_layers(self, num_layers: int, optimizer_kwargs: Dict[str, Any]) -> None:
        """Freeze the initial layers of the model."""
        for param in self.classifier.model.parameters():
            param.requires_grad_(False)
        self.classifier._reset_optimizer(optimizer_kwargs)
