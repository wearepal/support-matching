from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
from typing_extensions import override

from ranzen.torch import DcModule
from torch import Tensor
import torch.nn as nn

__all__ = [
    "AeFactory",
    "AePair",
]

E = TypeVar("E", bound=nn.Module)
D = TypeVar("D", bound=nn.Module)


@dataclass(repr=False, eq=False)
class AePair(DcModule, Generic[E, D]):
    encoder: E
    decoder: D
    latent_dim: int

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.encoder(x)

    def encode(self, x: Tensor) -> Tensor:  # type: ignore
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:  # type: ignore
        return self.decoder(z)

    def encode_decode(self, x: Tensor) -> Tensor:  # type: ignore
        return self.encode(self.decoder(x))


@dataclass
class AeFactory(ABC):
    @abstractmethod
    def __call__(self, input_shape: tuple[int, int, int]) -> AePair:
        raise NotImplementedError()
