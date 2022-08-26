from enum import Enum
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ["Activation", "BiaslessLayerNorm"]


class Activation(Enum):
    RELU = partial(nn.ReLU)
    GELU = partial(nn.GELU)
    SELU = partial(nn.SELU)
    SWISH = SILU = partial(nn.SiLU)


class BiaslessLayerNorm(nn.Module):
    beta: Parameter

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gamma = Parameter(torch.ones(input_dim))
        self.register_buffer("beta", torch.zeros(input_dim))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.layer_norm(
            x,
            normalized_shape=x.shape[-1:],
            weight=self.gamma,
            bias=self.beta,
        )
