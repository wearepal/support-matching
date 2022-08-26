from typing import Any, Dict, Optional

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from ranzen.decorators import implements
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn

__all__ = ["Model"]


class Model(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_cls = optimizer_cls

        optimizer_config = DictConfig({"_target_": self.optimizer_cls})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)
        self.optimizer = instantiate(optimizer_config, params=self.parameters(), lr=self.lr)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> None:
        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
        else:
            self.optimizer.step()

    @implements(nn.Module)
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
