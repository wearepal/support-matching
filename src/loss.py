from enum import Enum
import math
from typing import Dict, List, Union

from conduit.types import Loss
from ranzen import str_to_enum
from ranzen.decorators import implements
from ranzen.torch.loss import ReductionType, cross_entropy_loss, reduce
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GeneralizedCELoss",
    "MixedLoss",
    "PolynomialLoss",
]


class MixedLoss(nn.Module):
    """Mix of cross entropy and MSE"""

    def __init__(
        self,
        feature_group_slices: Dict[str, List[slice]],
        *,
        disc_loss_factor: float = 1.0,
        reduction: Union[str, ReductionType] = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.reduction = str_to_enum(reduction, enum=ReductionType)
        assert feature_group_slices["discrete"][0].start == 0, "x has to start with disc features"
        self.disc_feature_slices = feature_group_slices["discrete"]
        assert all(group.stop - group.start >= 2 for group in self.disc_feature_slices)
        self.cont_start = self.disc_feature_slices[-1].stop
        self.disc_loss_factor = disc_loss_factor

    @implements(nn.Module)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # type: ignore
        disc_loss = input.new_zeros(())
        # for the discrete features do cross entropy loss
        for disc_slice in self.disc_feature_slices:
            disc_loss += cross_entropy_loss(
                input=input[:, disc_slice],
                target=target[:, disc_slice].argmax(dim=1),
                reduction=self.reduction,
            )
        # for the continuous features do MSE
        cont_loss = reduce(
            F.mse_loss(
                input=input[:, self.cont_start :],
                target=target[:, self.cont_start :],
                reduction="none",
            ),
            reduction_type=self.reduction,
        )
        return self.disc_loss_factor * disc_loss + cont_loss


class GeneralizedCELoss(nn.Module):
    def __init__(
        self,
        *,
        q: float = 0.7,
        reduction: Union[ReductionType, str] = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.q = q

    @implements(nn.Module)
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:  # type: ignore
        p = input.softmax(dim=1)
        p_correct = torch.gather(p, 1, torch.unsqueeze(target, 1))
        # modify gradient of cross entropy
        loss_weight = (p_correct.squeeze().detach() ** self.q) * self.q
        return cross_entropy_loss(
            input,
            target=target,
            reduction=self.reduction,
            instance_weight=loss_weight,
        )


class Mode(Enum):
    exp = "exponential"
    logit = "logit"
    linear = "linear"


class PolynomialLoss(nn.Module, Loss):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.

    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[mode]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    def __init__(
        self,
        mode: Union[str, Mode] = Mode.exp,
        *,
        alpha: float = 1.0,
        reduction: Union[str, ReductionType] = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.mode = str_to_enum(mode, enum=Mode)
        if (self.mode is Mode.linear) and (alpha <= 1):
            raise ValueError("'linear' mode requires 'alpha' to be greater than 1.")
        self.reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.alpha = alpha

    def margin_fn(self, margin_vals: torch.Tensor) -> Tensor:
        indicator = margin_vals <= 1
        inv_part = torch.pow(
            margin_vals.abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.mode is Mode.exp:
            exp_part = torch.exp(-1 * margin_vals)
            scores = exp_part * indicator + inv_part * (~indicator)
        if self.mode is Mode.logit:
            indicator = margin_vals <= 1
            inv_part = torch.pow(margin_vals.abs(), -1 * self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = F.softplus(logit_inner) / (math.log(1 + math.exp(-1)))
            scores = logit_part * indicator + inv_part * (~indicator)
        else:
            assert self.alpha > 1
            linear_part = -1 * margin_vals + torch.ones_like(margin_vals) * (
                self.alpha / (self.alpha - 1)
            )
            scores = linear_part * indicator + inv_part * (~indicator) / (self.alpha - 1)
        return scores

    @implements(nn.Module)
    def forward(self, input: Tensor, *, target: Tensor, instance_weight: Optional[Tensor] = None) -> Tensor:  # type: ignore
        dim = input.size(1)
        if dim > 2:
            raise ValueError(
                "PolynomialLoss is only applicable to binary classification: logits must be of size"
                " 1 or 2 at dimension 1."
            )
        elif dim == 1:
            input = torch.cat((1 - input, input), dim=1)
        target_sign = 2 * target - 1  # y \in {0, 1} -> y \in {-1, 1}
        margin_scores = (input[:, 1] - input[:, 0]) * target_sign
        losses = self.margin_fn(margin_scores)
        if instance_weight is not None:
            losses *= instance_weight.view_as(losses)
        return reduce(losses, reduction_type=self.reduction)
