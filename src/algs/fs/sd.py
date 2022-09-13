from dataclasses import dataclass, field
from typing import List, Tuple, Union

from conduit.types import Loss
from omegaconf.listconfig import ListConfig
from ranzen import implements, str_to_enum
from ranzen.torch.loss import CrossEntropyLoss, ReductionType, reduce  # type: ignore
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
        lambda_: Union[float, Tuple[float, ...], List[float], ListConfig] = 1.0,
        *,
        gamma: Union[float, Tuple[float, ...], List[float], ListConfig] = 0.0,
        reduction: Union[str, ReductionType] = ReductionType.mean,
    ) -> None:
        super().__init__()
        if isinstance(reduction, str):
            reduction = str_to_enum(str_=reduction, enum=ReductionType)
        if isinstance(lambda_, ListConfig):
            lambda_ = list(lambda_)
        if isinstance(gamma, ListConfig):
            gamma = list(gamma)
        self.loss_fn = CrossEntropyLoss(reduction=reduction)
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
    def reduction(self, value: Union[ReductionType, str]) -> None:
        self.loss_fn.reduction = value

    @implements(nn.Module)
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:  # type: ignore
        lambda_ = self.lambda_[target] if isinstance(self.lambda_, Tensor) else self.lambda_
        loss = self.loss_fn(input, target=target)
        reg = 0.5 * reduce(
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
            reduction=ReductionType.mean,
        )
        super().__post_init__()
