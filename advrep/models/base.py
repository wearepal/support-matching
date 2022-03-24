from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

from ranzen.decorators import implements
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributions as td
import torch.nn as nn
from torch.optim import Adam

__all__ = [
    "EncodingSize",
    "ModelBase",
    "Reconstructions",
    "SplitDistributions",
    "SplitEncoding",
    "replace_zs",
]


class EncodingSize(NamedTuple):
    zs: int
    zy: int


@dataclass
class SplitEncoding:
    zs: Tensor
    zy: Tensor

    def join(self) -> Tensor:
        return torch.cat([self.zs, self.zy], dim=1)


def replace_zs(enc: SplitEncoding, new_zs: Tensor) -> SplitEncoding:
    return SplitEncoding(zs=new_zs, zy=enc.zy)


class SplitDistributions(NamedTuple):
    zs: td.Distribution
    zy: td.Distribution


class Reconstructions(NamedTuple):
    all: Tensor
    rand_s: Tensor  # reconstruction with random s
    rand_y: Tensor  # reconstruction with random y
    zero_s: Tensor
    zero_y: Tensor
    just_s: Tensor


class ModelBase(nn.Module):

    default_kwargs = dict(optimizer_kwargs=dict(lr=1e-3, weight_decay=0))

    def __init__(self, model: nn.Module, *, optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model = model
        optimizer_kwargs = optimizer_kwargs or self.default_kwargs["optimizer_kwargs"]
        self.optimizer = Adam(self.model.parameters(), **optimizer_kwargs)

    def reset_parameters(self) -> None:
        def _reset_parameters(m: nn.Module):
            if hasattr(m.__class__, "reset_parameters") and callable(
                getattr(m.__class__, "reset_parameters")
            ):
                m.reset_parameters()  # type: ignore

        self.model.apply(_reset_parameters)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> None:
        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
        self.optimizer.step()

    @implements(nn.Module)
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
