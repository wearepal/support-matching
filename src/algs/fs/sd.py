from dataclasses import dataclass, field
from typing import Tuple, Union

from conduit.types import Loss
from ranzen import implements
from ranzen.torch.loss import CrossEntropyLoss, ReductionType, _reduce  # type: ignore
import torch
from torch import Tensor
import torch.nn as nn

from .erm import Erm

__all__ = [
    "SdErm",
    "SdRegularisedXent",
]


class SdRegularisedXent(nn.Module, Loss):
    """Cross-entropy loss with spectral-decoupling."""

    lambda_: Union[float, Tensor]

    def __init__(
        self,
        lambda_: Union[float, Tuple[float, ...]] = 1.0,
        *,
        gamma: Union[float, Tuple[float, ...]] = 0.0,
        reduction_type: ReductionType = ReductionType.mean,
    ) -> None:
        self.loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)
        if isinstance(lambda_, tuple):
            self.register_buffer("lambda_", torch.as_tensor(lambda_, dtype=torch.float))
        else:
            self.lambda_ = lambda_
        if isinstance(gamma, tuple):
            self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float).unsqueeze(0))
        else:
            self.gamma = gamma

    @property
    def reduction(self) -> Union[ReductionType, str]:
        return self.loss_fn.reduction

    @reduction.setter
    def reduction(self, value: Union[ReductionType, str]) -> None:
        self.loss_fn.reduction = value

    @implements(nn.Module)
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:  # type: ignore
        lambda_ = self.lambda_[target] if isinstance(self.lambda_, Tensor) else self.lambda_
        loss = self.loss_fn(input, target=target)
        reg = 0.5 * _reduce(
            lambda_ * (input - self.gamma).square().sum(dim=1),
            reduction_type=self.reduction,
        )
        return loss + reg


@dataclass(eq=False)
class SdErm(Erm):
    """ERM with spectral decoupling applied to the logits, as proposed in `Gradient Starvation`_
    .. _Gradient Starvation:
        https://arxiv.org/abs/2011.09468
    """

    criterion: SdRegularisedXent = field(init=False)
    lambda_: Union[float, Tuple[float, ...]] = 1.0
    gamma: Union[float, Tuple[float, ...]] = 0.0

    def __post_init__(self) -> None:
        self.criterion = SdRegularisedXent(
            lambda_=self.lambda_,
            gamma=self.gamma,
            reduction_type=ReductionType.mean,
        )
        super().__post_init__()
