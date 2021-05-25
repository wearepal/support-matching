from __future__ import annotations
import logging
from typing import Any, Callable, cast

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import trange

from suds.models import Classifier

__all__ = ["GDRO"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class LossComputer(nn.Module):
    def __init__(
        self,
        criterion: Callable[[Tensor, Tensor], Tensor],
        is_robust: bool,
        group_counts: Tensor,
        alpha: np.ndarray | None = None,
        gamma: float = 0.1,
        adj: Tensor | None = None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        super().__init__()
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

    def forward(self, yhat: Tensor, y, group_idx: Tensor | None = None) -> Tensor:
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss: Tensor, group_count: Tensor) -> tuple[Tensor, Tensor]:
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(
        self, group_loss: Tensor, group_count: Tensor
    ) -> tuple[Tensor, Tensor]:
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

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
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
        if self.is_robust:
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

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict["model_norm_sq"] = model_norm_sq.item()
        stats_dict["reg_loss"] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f"avg_loss_group:{idx}"] = self.avg_group_loss[idx].item()
            stats_dict[f"exp_avg_loss_group:{idx}"] = self.exp_avg_loss[idx].item()
            stats_dict[f"avg_acc_group:{idx}"] = self.avg_group_acc[idx].item()
            stats_dict[f"processed_data_count_group:{idx}"] = self.processed_data_counts[idx].item()
            stats_dict[f"update_data_count_group:{idx}"] = self.update_data_counts[idx].item()
            stats_dict[f"update_batch_count_group:{idx}"] = self.update_batch_counts[idx].item()

        stats_dict["avg_actual_loss"] = self.avg_actual_loss.item()
        stats_dict["avg_per_sample_loss"] = self.avg_per_sample_loss.item()
        stats_dict["avg_acc"] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict


class GDRO(Classifier):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] | None,
        group_counts: Tensor,
        alpha: np.ndarray | None = None,
        normalize_loss: bool = False,
        gamma: float = 0.1,
        step_size: float = 0.01,
        btl: bool = False,
    ):
        super().__init__(model, num_classes, optimizer_kwargs=optimizer_kwargs, criterion="ce")
        # TODO: criterion should depend on the number of classes
        self.loss_computer = LossComputer(
            criterion=nn.CrossEntropyLoss(reduction="none"),
            is_robust=True,
            group_counts=group_counts,
            alpha=alpha,
            normalize_loss=normalize_loss,
            gamma=gamma,
            step_size=step_size,
            btl=btl,
        )

    def fit(
        self,
        train_data: Dataset | DataLoader,
        epochs: int,
        device: torch.device,
        test_data: Dataset | DataLoader | None = None,
        pred_s: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        **train_loader_kwargs: dict[str, Any],
    ):
        del pred_s
        if not isinstance(train_data, DataLoader):
            # Default settings for train-loader
            train_loader_kwargs.setdefault("pin_memory", True)  # type: ignore
            train_loader_kwargs.setdefault("shuffle", True)  # type: ignore

            train_data = DataLoader(
                train_data,
                batch_size=batch_size,
                **train_loader_kwargs,
            )
        if test_data is not None:
            if not isinstance(test_data, DataLoader):
                test_data = DataLoader(
                    test_data,
                    batch_size=test_batch_size,
                    shuffle=False,
                    pin_memory=train_data.pin_memory,
                    num_workers=train_data.num_workers,
                )
        LOGGER.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()

            for inputs, group_idx, targets in train_data:
                inputs, group_idx, targets = self.to_device(
                    inputs, group_idx, targets, device=device
                )
                group_idx.squeeze_()
                targets.squeeze_()

                self.optimizer.zero_grad()
                loss, _ = self.routine(data=inputs, targets=targets, group_idx=group_idx)
                loss.backward()
                self.step()

            if test_data is not None:
                self.eval()
                sum_test_acc = 0.0
                num_samples = 0

                with torch.set_grad_enabled(False):
                    for inputs, group_idx, targets in test_data:
                        inputs, group_idx, targets = self.to_device(
                            inputs, group_idx, targets, device=device
                        )
                        group_idx.squeeze_()
                        targets.squeeze_()

                        loss, acc = self.routine(inputs, targets, group_idx=group_idx)
                        sum_test_acc += acc * targets.size(0)
                        num_samples += targets.size(0)  # undo the batch-wise averaging
                avg_test_acc = sum_test_acc / num_samples

                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()

    def routine(
        self,
        data: Tensor,
        targets: Tensor,
        group_idx: Tensor,
        instance_weights: Tensor | None = None,
    ) -> tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """

        yhat = self.model(data).view(-1)
        loss = self.loss_computer(yhat=yhat, y=targets, group_idx=group_idx)
        acc = self.compute_accuracy(yhat, targets)

        return loss, acc
