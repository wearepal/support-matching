from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional
from typing_extensions import override

from conduit.types import LRScheduler
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from ranzen.torch import DcModule
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

from .utils import exclude_from_weight_decay

__all__ = ["Model", "ModelCfg", "Optimizer"]


class Optimizer(Enum):
    ADAM = torch.optim.AdamW
    RADAM = torch.optim.RAdam


@dataclass
class ModelCfg:
    """These are the parameters to `Model` which are configurable by hydra."""

    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[dict] = None
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None


@dataclass(repr=False, eq=False)
class Model(DcModule):
    _PBAR_COL: ClassVar[str] = "#ffe252"

    model: nn.Module
    cfg: ModelCfg
    optimizer: torch.optim.Optimizer = field(init=False)
    scheduler: Optional[LRScheduler] = field(init=False, default=None)

    def __post_init__(self) -> None:
        optimizer_config = DictConfig({"weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr})
        if self.cfg.optimizer_kwargs is not None:
            optimizer_config.update(self.cfg.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        self.optimizer = self.cfg.optimizer_cls.value(**optimizer_config, params=params)
        if self.cfg.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.cfg.scheduler_cls})
            if self.cfg.scheduler_kwargs is not None:
                scheduler_config.update(self.cfg.scheduler_kwargs)
            self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

    def step(self, grad_scaler: Optional[GradScaler] = None, scaler_update: bool = True) -> None:
        if grad_scaler is None:
            self.optimizer.step()
        else:
            grad_scaler.step(self.optimizer)
            if scaler_update:
                grad_scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    @override
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
