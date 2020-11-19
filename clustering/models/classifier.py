import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from clustering.models.base import ModelBase
from shared.utils import wandb_log

__all__ = ["Classifier", "Regressor"]

log = logging.getLogger(__name__.split(".")[-1].upper())


class Classifier(ModelBase):
    """Wrapper for classifier models."""

    def __init__(self, model: nn.Module, num_classes: int, optimizer_kwargs: Optional[Dict] = None):
        """Build classifier model.

        Args:).
            n_classes: Positive integer. Number of class labels.
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        if num_classes < 2:
            raise ValueError(
                f"Invalid number of classes: must equal 2 or more," f" {num_classes} given."
            )
        self.criterion = "bce" if num_classes == 2 else "ce"
        self.num_classes = num_classes

        super().__init__(model, optimizer_kwargs=optimizer_kwargs)

    def apply_criterion(self, logits: Tensor, targets: Tensor) -> Tensor:
        # if self.criterion == "bce":
        #     if targets.dtype != torch.float32:
        #         targets = targets.float()
        #     logits = logits.view(-1, 1)
        #     targets = targets.view(-1, 1)
        #     return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # else:
        targets = targets.view(-1)
        if targets.dtype != torch.long:
            targets = targets.long()
        return F.cross_entropy(logits, targets, reduction="none")

    def predict(self, inputs: Tensor, top: int = 1) -> Tensor:
        """Make prediction.

        Args:
            inputs: Tensor. Inputs to the classifier.
            top: Int. Top-k accuracy.

        Returns:
            Class predictions (tensor) for the given data samples.
        """
        outputs = super().__call__(inputs)
        # if self.criterion == "bce":
        #     pred = torch.round(outputs.sigmoid())
        # else:
        _, pred = outputs.topk(top, 1, True, True)

        return pred

    def predict_dataset(
        self, data: Union[Dataset, DataLoader], device: torch.device, batch_size: int = 100
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not isinstance(data, DataLoader):
            data = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        preds, actual, sens = [], [], []
        with torch.set_grad_enabled(False):
            for x, s, y in data:
                x = x.to(device)
                y = y.to(device)

                batch_preds = self.predict(x)
                preds.append(batch_preds)
                actual.append(y)
                sens.append(s)

        preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
        actual = torch.cat(actual, dim=0).cpu().detach().view(-1)
        sens = torch.cat(sens, dim=0).cpu().detach().view(-1)

        return preds, actual, sens

    def compute_accuracy(self, outputs: Tensor, targets: Tensor, top: int = 1) -> float:
        """Computes the classification accuracy.

        Args:
            outputs: Tensor. Classifier outputs.
            targets: Tensor. Targets for each input.
            top (int): Top-K accuracy.

        Returns:
            Accuracy of the predictions (float).
        """

        # if self.criterion == "bce":
        #     pred = torch.round(outputs.sigmoid())
        # else:
        _, pred = outputs.topk(top, 1, True, True)
        pred = pred.t().to(targets.dtype)
        correct = pred.eq(targets.view(1, -1).expand_as(pred)).float()
        correct = correct[:top].view(-1).float().sum(0, keepdim=True)
        accuracy = correct / targets.size(0) * 100

        return accuracy.detach().item()

    def routine(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        outputs = super().__call__(data)
        loss = self.apply_criterion(outputs, targets)
        loss = loss.mean()
        acc = self.compute_accuracy(outputs, targets)

        return loss, acc

    def fit(
        self,
        train_data: Union[Dataset, DataLoader],
        epochs: int,
        device: torch.device,
        use_wandb: bool,
        test_data: Optional[Union[Dataset, DataLoader]] = None,
        pred_s: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        lr_milestones: Optional[Dict] = None,
    ) -> None:
        if not isinstance(train_data, DataLoader):
            train_data = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, pin_memory=True
            )
        if test_data is not None:
            if not isinstance(test_data, DataLoader):
                test_data = DataLoader(
                    test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True
                )

        scheduler = None
        if lr_milestones is not None:
            scheduler = MultiStepLR(optimizer=self.optimizer, **lr_milestones)

        log.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()
            for step, (x, *target) in enumerate(train_data, start=epoch * len(train_data)):
                if len(target) == 2:
                    target = target[0] if pred_s else target[1]
                else:
                    target = target[0]

                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                loss, acc = self.routine(x, target)
                loss.backward()
                self.optimizer.step()
                wandb_log(use_wandb, {"loss": loss.item()}, step=step)
                pbar.set_postfix(epoch=epoch + 1, train_loss=loss.item(), train_acc=acc)

            if test_data is not None:

                self.model.eval()
                avg_test_acc = 0.0

                with torch.set_grad_enabled(False):
                    for x, s, y in test_data:

                        if pred_s:
                            target = s
                        else:
                            target = y

                        x = x.to(device)
                        target = target.to(device)

                        loss, acc = self.routine(x, target)
                        avg_test_acc += acc

                avg_test_acc /= len(test_data)

                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

            if scheduler is not None:
                scheduler.step(epoch)
        pbar.close()


class Regressor(Classifier):
    """Wrapper for regression models."""

    def __init__(
        self, model: nn.Module, optimizer_kwargs: Optional[Dict[str, float]] = None
    ) -> None:
        """Build classifier model.

        Args:
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        super().__init__(model, 2, optimizer_kwargs=optimizer_kwargs)
        self.criterion = "mse"

    def apply_criterion(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(logits, targets.flatten(start_dim=1), reduction="none")

    def predict(self, inputs: Tensor, top: int = 1) -> Tensor:
        """Make prediction."""
        return super().__call__(inputs)

    def compute_accuracy(self, outputs: Tensor, targets: Tensor, top: int = 1) -> float:
        return 0
