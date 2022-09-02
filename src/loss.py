from typing import Dict, List, Union
from typing_extensions import Literal

from ranzen import str_to_enum
from ranzen.decorators import implements
from ranzen.torch import ReductionType, cross_entropy_loss
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GeneralizedCELoss",
    "MixedLoss",
]


class MixedLoss(nn.Module):
    """Mix of cross entropy and MSE"""

    def __init__(
        self,
        feature_group_slices: Dict[str, List[slice]],
        disc_loss_factor: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        assert feature_group_slices["discrete"][0].start == 0, "x has to start with disc features"
        self.disc_feature_slices = feature_group_slices["discrete"]
        assert all(group.stop - group.start >= 2 for group in self.disc_feature_slices)
        self.cont_start = self.disc_feature_slices[-1].stop
        self.disc_loss_factor = disc_loss_factor
        self.reduction = reduction

    @implements(nn.Module)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # type: ignore
        disc_loss = input.new_zeros(())
        # for the discrete features do cross entropy loss
        for disc_slice in self.disc_feature_slices:
            disc_loss += F.cross_entropy(
                input[:, disc_slice], target[:, disc_slice].argmax(dim=1), reduction=self.reduction
            )
        # for the continuous features do MSE
        cont_loss = F.mse_loss(
            input[:, self.cont_start :], target[:, self.cont_start :], reduction=self.reduction
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
        if isinstance(reduction, str):
            reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.reduction = reduction
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
