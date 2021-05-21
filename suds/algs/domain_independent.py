from __future__ import annotations
import logging
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from suds.models.classifier import Classifier

__all__ = ["DomainIndependentClassifier"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class DomainIndependentClassifier(Classifier):
    """
    Implementation of the "Domain Independent Architecture" described in the paper
    'Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation'.

    A separate classification layer is trained for each domain on top of a shared representation.
    At test time, predictions are made by summing over the activations of the classifier heads.
    For linear classifiers with a shared feature representation this corresponds to averaging the
    class decision boundaries.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        num_domains: int,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        """Build classifier model.

        Args:).
            n_classes: Positive integer. Number of class labels.
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        super().__init__(model, optimizer_kwargs=optimizer_kwargs, num_classes=num_classes)
        self.num_domains = num_domains

    def _unroll_logits(self, logits: Tensor) -> Tensor:
        return logits.view(logits.size(0), -1, self.num_domains)

    def apply_criterion(self, logits: Tensor, targets: Tensor, domain_labels: Tensor) -> Tensor:
        logits_unrolled = self._unroll_logits(logits)
        logits_selected = logits_unrolled.gather(
            -1, domain_labels.view(-1, 1, 1).expand(-1, logits_unrolled.size(1), -1)
        ).squeeze(-1)
        return super().apply_criterion(logits_selected, targets)

    def _inference_sum_out(self, logits: Tensor, top: int = 1) -> Tensor:
        """Inference method: sum the output across domains."""
        logits_summed = self._unroll_logits(logits).sum(-1)
        #
        if logits.size(1) == 1:  # Binary classification
            return logits_summed > 0
        return logits.topk(k=top, dim=1).indices

    def predict(self, inputs: Tensor, top: int = 1) -> Tensor:
        """Make prediction.

        Args:
            inputs: Tensor. Inputs to the classifier.
            top: Int. Top-k accuracy.

        Returns:
            Class predictions (tensor) for the given data samples.
        """
        logits = self.model(inputs)
        return self._inference_sum_out(logits, top=top)

    def predict_dataset(
        self, data: Dataset | DataLoader, device: torch.device, batch_size: int = 100
    ):
        if not isinstance(data, DataLoader):
            data = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        preds, actual, sens = [], [], []
        with torch.set_grad_enabled(False):
            for x, s, y in data:
                x, s, y = self.to_device(x, s, y, device=device)

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
        pred = self._inference_sum_out(outputs, top=top)
        pred = pred.t().to(targets.dtype)
        correct = pred.eq(targets.view(1, -1).expand_as(pred)).float()
        correct = correct[:top].view(-1).float().sum(0, keepdim=True)
        accuracy = correct / targets.size(0) * 100

        return accuracy.detach().item()  # type: ignore

    def routine(
        self,
        data: Tensor,
        targets: Tensor,
        domain_labels: Tensor,
        instance_weights: Tensor | None = None,
    ) -> tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        logits = self.model(data)
        loss = self.apply_criterion(logits, targets, domain_labels=domain_labels)
        if instance_weights is not None:
            loss = loss.view(-1) * instance_weights.view(-1)
        loss = loss.mean()
        acc = self.compute_accuracy(logits, targets)

        return loss, acc

    def fit(
        self,
        train_data: Dataset | DataLoader,
        epochs: int,
        device: torch.device,
        test_data: Dataset | DataLoader | None = None,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        **train_loader_kwargs: dict[str, Any],
    ):

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
                x, s, y = self.to_device(x, s, y, device=device)

                self.optimizer.zero_grad()
                loss, acc = self.routine(x, targets=y, domain_labels=s)
                loss.backward()
                self.step()

            if test_data is not None:

                self.model.eval()
                sum_test_acc = 0.0
                # We could just obtain this count using len(dataloader.dataset) but then
                # the type-checker complains because a Dataset object doesn't have to implement
                # __len__ (it makes no sense for iterable datasets, for instance)
                num_samples = 0
                with torch.no_grad():
                    for x, s, y in test_data:
                        x, s, y = self.to_device(x, s, y, device=device)
                        loss, acc = self.routine(x, targets=y, domain_labels=s)
                        sum_test_acc += acc * y.size(0)  # undo the batch-wise averaging
                        num_samples += y.size(0)
                avg_test_acc = sum_test_acc / num_samples
                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()
