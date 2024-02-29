from collections.abc import Callable
from enum import Enum

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ["Activation", "BiaslessLayerNorm"]


class Activation(Enum):
    RELU = (nn.ReLU,)
    GELU = (nn.GELU,)
    SELU = (nn.SELU,)
    SWISH = SILU = (nn.SiLU,)

    def __init__(self, init: Callable[..., nn.Module]) -> None:
        self.init = init


class BiaslessLayerNorm(nn.Module):
    beta: Parameter

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gamma = Parameter(torch.ones(input_dim))
        self.register_buffer("beta", torch.zeros(input_dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, normalized_shape=x.shape[-1:], weight=self.gamma, bias=self.beta)


class NormType(Enum):
    BN = (nn.BatchNorm1d,)
    LN = (BiaslessLayerNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init
