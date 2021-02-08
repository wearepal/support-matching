from __future__ import annotations
import logging
from math import inf
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import trange

from fdm.models import Classifier
from fdm.optimisation.utils import ExtractableDataset, extract_labels_from_dataset

__all__ = ["GDRO"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class GDRO(Classifier):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] | None,
        c_param: float = 1.0,
    ):
        super().__init__(model, num_classes, optimizer_kwargs=optimizer_kwargs, criterion="ce")
        self.c_param = c_param

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
        LOGGER.info("Training classifier...")
        unique_s = extract_labels_from_dataset(train_data)[0].unique()
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()

            for x, s, y in train_loader:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                loss = []
                for _s in unique_s:
                    if (s == _s).sum().gt(0).item():
                        _loss, _ = self._routine(x[s == _s], y[s == _s])
                    else:
                        _loss = -torch.tensor(inf).to(x.device)
                    loss.append(_loss)

                max(loss).backward()
                self.step()

            if test_loader is not None:
                self.eval()
                avg_test_acc = 0.0

                with torch.set_grad_enabled(False):
                    for x, s, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        loss, acc = self.routine(x, y)
                        avg_test_acc += acc

                avg_test_acc /= len(test_loader)

                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()

    def _routine(
        self,
        data: Tensor,
        targets: Tensor,
        instance_weights: Tensor | None = None,
    ) -> tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        outputs = self(data)
        loss = self.apply_criterion(outputs, targets)
        if instance_weights is not None:
            loss = loss.view(-1) * instance_weights.view(-1)
        loss = loss.mean()
        loss += self.c_param / torch.sqrt(torch.ones_like(loss) * data.shape[0])
        acc = self.compute_accuracy(outputs, targets)

        return loss, acc
