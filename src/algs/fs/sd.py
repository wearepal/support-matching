from dataclasses import dataclass, field
from typing import Optional, Union
from typing_extensions import override

from conduit.types import Loss
from omegaconf.listconfig import ListConfig
from ranzen.torch.loss import CrossEntropyLoss, ReductionType, reduce
import torch
from torch import Tensor
import torch.nn as nn

from .erm import Erm

__all__ = ["SdErm", "SdCrossEntropyLoss"]


class SdCrossEntropyLoss(nn.Module, Loss):
    """Cross-entropy loss with spectral-decoupling."""

    lambda_: Union[float, Tensor]
    gamma: Union[float, Tensor]

    def __init__(
        self,
        loss_fn: Optional[Loss] = None,
        *,
        lambda_: Union[float, tuple[float, ...], list[float], ListConfig] = 1.0,
        gamma: Union[float, tuple[float, ...], list[float], ListConfig] = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(lambda_, ListConfig):
            lambda_ = list(lambda_)
        if isinstance(gamma, ListConfig):
            gamma = list(gamma)
        if loss_fn is None:
            cross_entropy: Loss = CrossEntropyLoss(reduction=ReductionType.mean)  # type: ignore
            loss_fn = cross_entropy
        self.loss_fn = loss_fn
        if isinstance(lambda_, (tuple, list)):
            self.register_buffer("lambda_", torch.as_tensor(lambda_, dtype=torch.float))
        else:
            self.lambda_ = lambda_
        if isinstance(gamma, (tuple, list)):
            self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float).unsqueeze(0))
        else:
            self.gamma = gamma

    @property
    def reduction(self) -> Union[ReductionType, str]:
        return self.loss_fn.reduction

    @reduction.setter
    def reduction(self, value: Union[ReductionType, str]) -> None:  # type: ignore
        self.loss_fn.reduction = value

    @override
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:  # type: ignore
        lambda_ = self.lambda_[target] if isinstance(self.lambda_, Tensor) else self.lambda_
        loss = self.loss_fn(input, target=target)
        reg = 0.5 * reduce(
            lambda_ * (input - self.gamma).square().sum(dim=1),
            reduction_type=self.reduction,
        )
        return loss + reg


@dataclass(repr=False, eq=False)
class SdErm(Erm):
    """ERM with spectral decoupling applied to the logits, as proposed in `Gradient Starvation`_
    .. _Gradient Starvation:
        https://arxiv.org/abs/2011.09468
    """

    lambda_: list[float] = field(default_factory=lambda: [1.0])
    gamma: list[float] = field(default_factory=lambda: [0.0])

    def __post_init__(self) -> None:
        self.criterion: SdCrossEntropyLoss = SdCrossEntropyLoss(
            lambda_=self.lambda_, gamma=self.gamma
        )
        super().__post_init__()
