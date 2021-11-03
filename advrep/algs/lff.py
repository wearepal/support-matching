from __future__ import annotations
import copy
import logging
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.std import trange

from advrep.models import Classifier
from advrep.models.base import ModelBase
from advrep.optimisation import (
    ExtractableDataset,
    GeneralizedCELoss,
    extract_labels_from_dataset,
)

__all__ = ["LfF"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


class EMA:
    def __init__(self, labels: Tensor, device: torch.device, alpha=0.9):
        self.labels = labels.view(-1).to(device)
        self.alpha = alpha
        self.parameter = torch.zeros(labels.size(0), device=device)
        self.updated = torch.zeros(labels.size(0), device=device)

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    def max_loss(self, label: int) -> Tensor:
        label_index = self.labels == label
        return self.parameter[label_index].max()


class LfF(Classifier):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] | None = None,
        q: float = 0.7,
    ):
        super().__init__(
            model=model,
            num_classes=num_classes,
            criterion="ce",
            optimizer_kwargs=optimizer_kwargs,
        )
        self.biased_model = ModelBase(copy.deepcopy(self.model), optimizer_kwargs=optimizer_kwargs)
        self.biased_criterion = GeneralizedCELoss(q=q)

    def fit(
        self,
        train_data: ExtractableDataset,
        epochs: int,
        device: torch.device,
        test_data: Dataset | None = None,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        **train_loader_kwargs: dict[str, Any],
    ):
        _, y = extract_labels_from_dataset(train_data)  # type: ignore
        train_data = IndexDataset(train_data)  # type: ignore

        # Default settings for train-loader
        train_loader_kwargs.setdefault("pin_memory", True)  # type: ignore
        train_loader_kwargs.setdefault("shuffle", True)  # type: ignore

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            **train_loader_kwargs,
        )
        test_loader = None
        if test_data is not None:
            test_loader = DataLoader(
                test_data,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=train_loader.pin_memory,
                num_workers=train_loader.num_workers,
            )

        sample_loss_ema_b = EMA(y.long(), alpha=0.7, device=device)
        sample_loss_ema_d = EMA(y.long(), alpha=0.7, device=device)

        LOGGER.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()
            for index, x, _, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(-1)

                logit_b = self.biased_model(x)
                logit_d = self.model(x)
                loss_b = self.apply_criterion(logit_b, y).detach()
                loss_d = self.apply_criterion(logit_d, y).detach()

                # EMA sample loss
                sample_loss_ema_b.update(loss_b.view(-1), index)
                sample_loss_ema_d.update(loss_d.view(-1), index)

                # class-wise normalize
                loss_b = sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = sample_loss_ema_d.parameter[index].clone().detach()

                for c in range(logit_d.size(1)):
                    class_index = y == c
                    max_loss_b = sample_loss_ema_b.max_loss(c)
                    max_loss_d = sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d

                # re-weighting based on loss value / generalized CE for biased model
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_b_update = self.biased_criterion(logit_b, y.long())
                loss_d_update = self.apply_criterion(logit_d, y) * loss_weight.to(device)
                loss = loss_b_update.mean() + loss_d_update.mean()

                self.zero_grad()
                self.biased_model.zero_grad()
                loss.backward()
                self.step()
                self.biased_model.step()

            if test_loader is not None:
                self.model.eval()
                sum_test_acc = 0.0
                num_samples = 0
                with torch.no_grad():
                    for x, _, y in test_loader:
                        x = x.to(device)
                        y = y.to(device).view(-1)
                        _, acc = self.routine(x, y)
                        sum_test_acc += acc * y.size(0)
                        num_samples += y.size(0)
                avg_test_acc = sum_test_acc / num_samples

                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()
