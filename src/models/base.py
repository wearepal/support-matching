from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar, Generic, Protocol, cast
from typing_extensions import TypeVar, override

from conduit.types import LRScheduler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

from .utils import DcModule, exclude_from_weight_decay

__all__ = ["Model", "Optimizer", "OptimizerCfg"]


class Optimizer(Enum):
    ADAM = torch.optim.AdamW
    RADAM = torch.optim.RAdam


@dataclass
class OptimizerCfg:
    """Configuration for an optimizer."""

    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: dict | None = None
    scheduler_cls: str | None = None
    scheduler_kwargs: dict | None = None


class ModuleLike(Protocol):
    def __call__(self, inputs: Tensor) -> Tensor:
        ...


M = TypeVar("M", bound=ModuleLike, default=nn.Module, covariant=True)


@dataclass(repr=False, eq=False)
class Model(DcModule, Generic[M]):
    _PBAR_COL: ClassVar[str] = "#ffe252"

    model: M
    opt: OptimizerCfg

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        optimizer_config = DictConfig({"weight_decay": self.opt.weight_decay, "lr": self.opt.lr})
        if self.opt.optimizer_kwargs is not None:
            optimizer_config.update(self.opt.optimizer_kwargs)

        params = exclude_from_weight_decay(
            self.named_parameters(), weight_decay=optimizer_config["weight_decay"]
        )
        kwargs = OmegaConf.to_container(optimizer_config, resolve=True)
        assert isinstance(kwargs, dict)
        return self.opt.optimizer_cls.value(**cast(dict[str, Any], kwargs), params=params)

    @cached_property
    def scheduler(self) -> LRScheduler | None:
        if self.opt.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.opt.scheduler_cls})
            if self.opt.scheduler_kwargs is not None:
                scheduler_config.update(self.opt.scheduler_kwargs)
            return instantiate(scheduler_config, optimizer=self.optimizer)
        return None

    def step(self, grad_scaler: GradScaler | None = None, scaler_update: bool = True) -> None:
        if grad_scaler is None:
            self.optimizer.step()
        else:
            grad_scaler.step(self.optimizer)
            if scaler_update:
                grad_scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    @override
    def forward(self, inputs: Tensor) -> Any:
        return self.model(inputs)
