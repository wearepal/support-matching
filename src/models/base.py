from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from conduit.types import LRScheduler
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from ranzen.decorators import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

from .utils import exclude_from_weight_decay

__all__ = ["Model", "Optimizer"]


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
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    scheduler: Optional[LRScheduler] = field(init=False, default=None)

    def __post_init__(self) -> None:
        optimizer_config = DictConfig({"weight_decay": self.weight_decay, "lr": self.lr})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        self.optimizer = self.optimizer_cls.value(**optimizer_config, params=params)
        if self.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if self.scheduler_kwargs is not None:
                scheduler_config.update(self.scheduler_kwargs)
            self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> None:
        if grad_scaler is None:
            self.optimizer.step()
        else:
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    @implements(DcModule)
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
