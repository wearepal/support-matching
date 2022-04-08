from typing import Any, Callable, Generic, Iterator, Optional
from typing_extensions import Concatenate, ParamSpec

from ranzen.decorators import implements
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.optim.optimizer import Optimizer

__all__ = ["Model"]

P = ParamSpec("P")


class Model(nn.Module, Generic[P]):
    def __init__(
        self,
        model: nn.Module,
        *args: P.args,
        lr: float = 5.0e-4,
        optimizer_cls: Callable[Concatenate[Iterator, float, P], Optimizer] = torch.optim.AdamW,
        **optimizer_kwargs: P.kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_cls = optimizer_cls

        self.optimizer = optimizer_cls(self.parameters(), self.lr, **optimizer_kwargs)

    def step(self, grad_scaler: Optional[GradScaler] = None) -> None:
        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
        else:
            self.optimizer.step()

    @implements(nn.Module)
    def forward(self, inputs: Tensor) -> Any:  # type: ignore
        return self.model(inputs)
