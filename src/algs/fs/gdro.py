from dataclasses import dataclass
from typing import Any, Optional
from typing_extensions import override

from conduit.data.structures import TernarySample
from conduit.types import Loss
import numpy as np
from ranzen.torch import cross_entropy_loss
from ranzen.torch.loss import ReductionType
import torch
from torch import Tensor
import torch.nn as nn

from src.data import DataModule, EvalTuple
from src.models import Classifier

from .base import FsAlg

__all__ = ["Gdro", "GdroClassifier", "LossComputer"]


class LossComputer(nn.Module):
    # Buffer modules
    group_counts: Tensor
    group_frac: Tensor
    adj: Tensor
    processed_data_counts: Tensor
    update_data_counts: Tensor
    update_batch_counts: Tensor
    avg_group_loss: Tensor
    avg_group_acc: Tensor

    def __init__(
        self,
        *,
        is_robust: bool,
        group_counts: Tensor,
        alpha: Optional[float] = None,
        gamma: float = 0.1,
        adj: Optional[Tensor] = None,
        min_var_weight: float = 0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
    ) -> None:
        super().__init__()
        self.reduction: ReductionType = ReductionType.mean
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.register_buffer("group_counts", group_counts)
        self.n_groups = len(group_counts)
        self.register_buffer("group_frac", self.group_counts / group_counts.sum())

        if adj is not None:
            adj = torch.from_numpy(adj).float()
        else:
            adj = torch.zeros(self.n_groups).float()
        self.register_buffer("adj", adj)

        if is_robust:
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.register_buffer("adv_probs", torch.ones(self.n_groups) / self.n_groups)
        self.register_buffer("exp_avg_loss", torch.zeros(self.n_groups))
        self.register_buffer("exp_avg_initialized", torch.zeros(self.n_groups).byte())

        self.register_buffer("processed_data_counts", torch.zeros(self.n_groups))
        self.register_buffer("update_data_counts", torch.zeros(self.n_groups))
        self.register_buffer("update_batch_counts", torch.zeros(self.n_groups))
        self.register_buffer("avg_group_loss", torch.zeros(self.n_groups))
        self.register_buffer("avg_group_acc", torch.zeros(self.n_groups))

        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    @staticmethod
    def _default_criterion(input: Tensor, *, target: Tensor) -> Tensor:
        return cross_entropy_loss(input=input, target=target, reduction=ReductionType.none)

    def forward(
        self, input: Tensor, *, target: Tensor, group_idx: Tensor, criterion: Optional[Loss] = None
    ) -> Tensor:  # type: ignore
        if criterion is None:
            per_sample_losses = self._default_criterion(input=input, target=target)
        else:
            if isinstance(red := criterion.reduction, ReductionType):
                assert red is ReductionType.none
            else:
                assert red == "none"
            # compute per-sample and per-group losses
            per_sample_losses = criterion(input=input, target=target)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx=group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(input, dim=1) == target).float(), group_idx
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(
            actual_loss=actual_loss,
            group_loss=group_loss,
            group_acc=group_acc,
            group_count=group_count,
            weights=weights,
        )

        return actual_loss

    def compute_robust_loss(self, group_loss: Tensor) -> tuple[Tensor, Tensor]:
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss: Tensor) -> tuple[Tensor, Tensor]:
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(
        self, group_loss: Tensor, ref_loss: Tensor
    ) -> tuple[Tensor, Tensor]:
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses: Tensor, group_idx: Tensor) -> tuple[Tensor, Tensor]:
        # compute observed counts and mean loss for each group
        group_map = (
            group_idx == torch.arange(self.n_groups, device=group_idx.device).unsqueeze(1).long()
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss: Tensor, group_count: Tensor) -> None:
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self) -> None:
        self.processed_data_counts.zero_()
        self.update_data_counts.zero_()
        self.update_batch_counts.zero_()
        self.avg_group_loss.zero_()
        self.avg_group_acc.zero_()
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(
        self,
        actual_loss: Tensor,
        *,
        group_loss: Tensor,
        group_acc: Tensor,
        group_count: Tensor,
        weights: Optional[Tensor] = None,
    ):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust and (weights is not None):
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


@dataclass(kw_only=True, repr=False, eq=False, frozen=True)
class GdroClassifier(Classifier):
    loss_computer: LossComputer

    def __post_init__(self) -> None:
        # LossComputer requires that the criterion return per-sample (unreduced) losses.
        if self.criterion is not None:
            self.criterion.reduction = ReductionType.none

    @override
    def training_step(self, batch: TernarySample[Tensor], *, pred_s: bool = False) -> Tensor:
        target = batch.s if pred_s else batch.y
        logits = self.forward(batch.x)
        return self.loss_computer.forward(
            input=logits, target=target, group_idx=batch.s, criterion=self.criterion
        )


@dataclass(repr=False, eq=False)
class Gdro(FsAlg):
    alpha: Optional[float] = 1.0
    normalize_loss: bool = False
    gamma: float = 0.1
    step_size: float = 0.01
    btl: bool = False
    adjustments: Optional[tuple[float]] = None
    criterion: Any = None  # Optional[Loss]

    @override
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        s_count = dm.card_s
        s_all = dm.train.s
        _, group_counts = s_all.unique(return_counts=True)

        adjustments = self.adjustments
        if adjustments is not None:
            assert len(adjustments) in (1, s_count)
            if len(adjustments) == 1:
                adjustments = np.array(adjustments * s_count)
            else:
                adjustments = np.array(adjustments)
        loss_computer = LossComputer(
            is_robust=True,
            group_counts=group_counts,
            alpha=self.alpha,
            normalize_loss=self.normalize_loss,
            gamma=self.gamma,
            step_size=self.step_size,
            btl=self.btl,
        )

        classifier = GdroClassifier(
            model=model, opt=self.opt, criterion=self.criterion, loss_computer=loss_computer
        )
        classifier.fit(
            train_data=dm.train_dataloader(),
            test_data=dm.test_dataloader(),
            steps=self.steps,
            val_interval=self.val_interval,
            device=self.device,
            grad_scaler=self.grad_scaler,
            use_wandb=True,
        )
        return classifier.predict(dm.test_dataloader(), device=self.device)
