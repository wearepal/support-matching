from collections.abc import Callable
from typing_extensions import override

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearResNet"]


class _LinearResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        planes: int,
        ctx_planes: int | None = None,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        zero_initialization: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(planes, eps=1e-3) for _ in range(2)]
            )
        if ctx_planes is not None:
            self.context_layer = nn.Linear(ctx_planes, planes)
        linear_layers_ls = [nn.Linear(planes, planes) for _ in range(2)]
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            nn.init.uniform_(linear_layers_ls[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(linear_layers_ls[-1].bias, -1e-3, 1e-3)
        self.linear_layers = nn.ModuleList(linear_layers_ls)

    def forward(self, inputs: Tensor, context: Tensor | None = None) -> Tensor:
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class LinearResNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        ctx_planes: int | None = None,
        num_blocks: int = 2,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.context_features = ctx_planes
        if ctx_planes is not None:
            self.initial_layer = nn.Linear(in_planes + ctx_planes, planes)
        else:
            self.initial_layer = nn.Linear(in_planes, planes)
        self.blocks = nn.ModuleList(
            [
                _LinearResidualBlock(
                    planes=planes,
                    ctx_planes=ctx_planes,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(planes, planes)

    @override
    def forward(self, inputs: Tensor, *, context: Tensor | None) -> Tensor:
        if context is not None:
            inputs = torch.cat((inputs, context), dim=1)
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps, context=context)
        return self.final_layer(temps)
