from torch import Tensor, nn

from .base import Encoder
from .classifier import Classifier

__all__ = ["SelfSupervised"]


class SelfSupervised(Classifier, Encoder):
    """Encoder trained with self-supervision."""

    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        return self.__call__(x)

    def get_encoder(self) -> nn.Module:
        encoder = self.model
        encoder.fc = nn.Identity()
        for param in encoder.parameters():
            param.requires_grad_(False)
        return encoder
