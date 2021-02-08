from __future__ import annotations
import logging
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import trange

from fdm.models import Classifier

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

            for x, s, y in train_data:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                loss = []
                for _s in s.unique():
                    _loss, _ = self._routine(x[s == _s], y[s == _s])
                    loss.append(_loss)

                max(loss).backward()
                self.step()

            if test_data is not None:
                self.eval()
                avg_test_acc = 0.0

                with torch.set_grad_enabled(False):
                    for x, s, y in test_data:
                        x = x.to(device)
                        y = y.to(device)

                        loss, acc = self.routine(x, y)
                        avg_test_acc += acc

                avg_test_acc /= len(test_data)

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
