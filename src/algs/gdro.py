from __future__ import annotations
from src.evaluation.metrics import compute_metrics, EvalPair
from conduit.data.structures import TernarySample
from typing_extensions import Self
from ranzen import gcopy
from src.data import DataModule, group_id_to_label
from dataclasses import dataclass, field
from .base import Algorithm
from typing import Callable, Optional, Tuple

from conduit.metrics import accuracy
import numpy as np
from ranzen import implements
from ranzen.torch.loss import ReductionType
import torch
from torch import Tensor
import torch.nn as nn

from omegaconf import DictConfig
from src.models import Classifier, Optimizer

__all__ = [
    "Gdro",
    "GdroClassifier",
    "LossComputer",
]


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
        criterion: Callable[[Tensor, Tensor], Tensor],
        is_robust: bool,
        group_counts: Tensor,
        alpha: float | None = None,
        gamma: float = 0.1,
        adj: Tensor | None = None,
        min_var_weight: float = 0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
    ) -> None:
        super().__init__()
        self.reduction: ReductionType = ReductionType.mean
        self.criterion = criterion
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

    def forward(self, input: Tensor, *, target: Tensor, group_idx: Tensor) -> Tensor:  # type: ignore
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(input, target)
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
        weights: Tensor | None = None,
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



@dataclass(eq=False)
class GdroClassifier(Classifier):
    criterion: LossComputer


    @implements(Classifier)
    def training_step(
        self,
        batch: TernarySample,
        *,
        pred_s: bool = False
    ) -> tuple[Tensor, float]:
        target = batch.s if pred_s else batch.y
        logits = self.forward(batch.x)
        loss = self.criterion(input=logits, target=target, group_idx=batch.y)
        loss = loss.mean()
        acc = accuracy(y_pred=logits, y_true=target).cpu().item()

        return loss, acc

@dataclass(eq=False)
class Gdro(Algorithm):
    alpha: Optional[float] = 1.0
    normalize_loss: bool = False
    gamma: float = 0.1
    step_size: float = 0.01
    btl: bool = False
    criterion: LossComputer = field(init=False)
    adjustments: Optional[Tuple[float]] = None
    steps: int = 10_000

    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[DictConfig] = None
    optimizer: torch.optim.Optimizer = field(init=False)


    @implements(Algorithm)
    def run(
        self, dm: DataModule, *, model: nn.Module) -> Self:
        dm = gcopy(dm, deep=False)
        s_count = dm.card_s
        if dm.deployment_ids is not None:
            y_dep = group_id_to_label(dm.deployment_ids, s_count=s_count, label="y").flatten()
            s_dep = group_id_to_label(dm.deployment_ids, s_count=s_count, label="s").flatten()
            dm.deployment.y = y_dep
            dm.deployment.s = s_dep
        elif dm.gt_deployment:
            dm.train += dm.deployment
        s_all, _ = dm.train.s
        group_counts = (torch.arange(s_count).unsqueeze(1) == s_all.squeeze()).sum(1).float()
        # process generalization adjustment stuff
        adjustments = self.adjustments
        if adjustments is not None:
            assert len(adjustments) in (1, s_count)
            if len(adjustments) == 1:
                adjustments = np.array(adjustments * s_count)
            else:
                adjustments = np.array(adjustments)
        self.criterion = LossComputer(
            criterion=nn.CrossEntropyLoss(reduction="none"),
            is_robust=True,
            group_counts=group_counts,
            alpha=self.alpha,
            normalize_loss=self.normalize_loss,
            gamma=self.gamma,
            step_size=self.step_size,
            btl=self.btl,
        )

        train_data = dm.train_dataloader()
        classifier = GdroClassifier(
            model=model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        classifier.fit(
            train_data=train_data,
            steps=self.steps,
            device=self.device,
            grad_scaler=self.grad_scaler,
        )

        # Generate predictions with the trained model
        preds, labels, sens = classifier.predict_dataset(dm.test_dataloader(), device=self.device)

        pair = EvalPair.from_tensors(y_pred=preds, y_true=labels, s=sens, pred_s=False)
        compute_metrics(
            pair=pair,
            model_name=self.__class__.__name__.lower(),
            step=0,
            s_dim=s_count,
            use_wandb=True,
        )
        return self