from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import override

from ranzen.torch import DcModule
from torch import Tensor
import torch.nn as nn

__all__ = ["AeFactory", "AePair"]


@dataclass(repr=False, eq=False)
class AePair(DcModule):
    encoder: nn.Module
    decoder: nn.Module
    latent_dim: int

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def encode_decode(self, x: Tensor) -> Tensor:
        return self.encode(self.decoder(x))


@dataclass
class AeFactory(ABC):
    @abstractmethod
    def __call__(self, input_shape: tuple[int, int, int]) -> AePair:
        raise NotImplementedError()
