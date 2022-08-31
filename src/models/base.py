from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from omegaconf.dictconfig import DictConfig
from ranzen.decorators import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

from .utils import exclude_from_weight_decay

__all__ = ["Model"]


class Optimizer(Enum):
    ADAM = torch.optim.AdamW
    RADAM = torch.optim.RAdam


@dataclass(eq=False)
class Model(DcModule):
    model: nn.Module
    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[DictConfig] = None
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:

        optimizer_config = DictConfig({"weight_decay": self.weight_decay, "lr": self.lr})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        self.optimizer = self.optimizer_cls.value(**optimizer_config, params=params)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> Optional[float]:
        if grad_scaler is None:
            return self.optimizer.step()
        return grad_scaler.step(self.optimizer)

    @implements(DcModule)
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
