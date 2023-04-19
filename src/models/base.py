from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional
from typing_extensions import override

from conduit.types import LRScheduler
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

from .utils import exclude_from_weight_decay

__all__ = ["Model", "ModelConf", "Optimizer"]


class Optimizer(Enum):
    ADAM = torch.optim.AdamW
    RADAM = torch.optim.RAdam


@dataclass
class ModelConf:
    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[Dict] = None
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[Dict] = None


class Model(nn.Module):
    _PBAR_COL: ClassVar[str] = "#ffe252"

    optimizer: torch.optim.Optimizer

    def __init__(self, cfg: ModelConf, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        optimizer_config = DictConfig({"weight_decay": cfg.weight_decay, "lr": cfg.lr})
        if cfg.optimizer_kwargs is not None:
            optimizer_config.update(cfg.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        self.optimizer = cfg.optimizer_cls.value(**optimizer_config, params=params)
        self.scheduler: Optional[LRScheduler] = None
        if cfg.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": cfg.scheduler_cls})
            if cfg.scheduler_kwargs is not None:
                scheduler_config.update(cfg.scheduler_kwargs)
            self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> None:
        if grad_scaler is None:
            self.optimizer.step()
        else:
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    @override
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
