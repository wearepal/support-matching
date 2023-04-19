from typing import Optional, Union
from typing_extensions import override

from attrs import define
from conduit.types import Loss
from ranzen import str_to_enum
from ranzen.torch.loss import CrossEntropyLoss, ReductionType, reduce
from torch import Tensor
import torch.nn as nn

from .erm import Erm

__all__ = ["DroLoss", "Dro"]


class DroLoss(nn.Module, Loss):
    """Fairness Without Demographics Loss."""

    def __init__(
        self,
        loss_fn: Optional[Loss] = None,
        *,
        eta: float = 0.5,
        reduction: Union[ReductionType, str] = ReductionType.mean,
    ) -> None:
        """Set up the loss, set which loss you want to optimize and the eta to offset by."""
        super().__init__()
        if isinstance(reduction, str):
            reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.reduction = reduction
        if loss_fn is None:
            loss_fn = CrossEntropyLoss(reduction=ReductionType.none)
        else:
            loss_fn.reduction = ReductionType.none
        self.reduction = reduction
        self.loss_fn = loss_fn
        self.eta = eta

    @override
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:  # type: ignore
        sample_losses = (self.loss_fn(input, target=target) - self.eta).relu().pow(2)
        return reduce(sample_losses, reduction_type=self.reduction)


@define(kw_only=True, repr=False, eq=False)
class Dro(Erm):
    eta: float = 0.5

    def __attrs_post_init__(self) -> None:
        self.criterion = DroLoss(self.criterion, eta=self.eta, reduction=ReductionType.mean)
        super().__attrs_post_init__()
