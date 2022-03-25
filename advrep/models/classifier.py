from __future__ import annotations
import logging
from typing import Any, Tuple, Union, overload
from typing_extensions import Literal

from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.types import Loss
from ranzen.torch.loss import CrossEntropyLoss
import torch
from torch import Tensor, nn
from tqdm import trange

from advrep.models.base import Model

__all__ = ["Classifier"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


@torch.no_grad()
def accuracy(logits: Tensor, *, targets: Tensor) -> float:
    logits = torch.atleast_2d(logits.squeeze())
    targets = torch.atleast_1d(targets.squeeze()).long()
    if len(logits) != len(targets):
        raise ValueError("'logits' and 'targets' must match in size at dimension 0.")
    preds = (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class Classifier(Model):
    """Wrapper for classifier models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: Loss | None = None,
    ) -> None:
        super().__init__(
            model, optimizer_cls=optimizer_cls, lr=lr, optimizer_kwargs=optimizer_kwargs
        )
        if criterion is None:
            criterion = CrossEntropyLoss()
        self.criterion = criterion

    def predict(self, inputs: Tensor) -> Tensor:
        logits = self.forward(inputs)
        return (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)

    def predict_soft(self, inputs: Tensor) -> Tensor:
        """Make soft predictions."""
        logits = self.forward(inputs)
        return logits.sigmoid() if logits.ndim == 1 else logits.softmax(dim=1)

    @overload
    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: Literal[False] = ...,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...

    @overload
    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: Literal[True],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...

    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        preds, actual, sens, soft_preds = [], [], [], []
        with torch.set_grad_enabled(False):
            for batch in data:
                batch.to(device)
                batch_preds = self.predict(batch.x)
                preds.append(batch_preds)
                actual.append(batch.y)
                sens.append(batch.s)
                if with_soft:
                    soft_preds.append(self.predict_soft(batch.x))

        preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
        actual = torch.cat(actual, dim=0).cpu().detach().view(-1)
        sens = torch.cat(sens, dim=0).cpu().detach().view(-1)
        soft_preds = torch.cat(soft_preds, dim=0).cpu().detach().view(-1)

        if with_soft:
            return preds, actual, sens, soft_preds
        else:
            return preds, actual, sens

    def training_step(
        self, input: Tensor, *, target: Tensor, instance_weights: Tensor | None = None
    ) -> tuple[Tensor, float]:
        logits = self.forward(input)
        loss = self.criterion(input=logits, target=target)
        if instance_weights is not None:
            loss = loss.view(-1) * instance_weights.view(-1)
        loss = loss.mean()
        acc = accuracy(logits=logits, targets=target)

        return loss, acc

    def fit(
        self,
        train_data: CdtDataLoader[TernarySample],
        *,
        epochs: int,
        device: torch.device,
        pred_s: bool = False,
        test_data: CdtDataLoader[TernarySample],
    ) -> None:
        LOGGER.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()

            for batch in train_data:
                batch = batch.to(device)
                target = batch.s if pred_s else batch.y

                self.optimizer.zero_grad()
                loss, acc = self.training_step(input=batch.x, target=target)
                loss.backward()
                self.step()

            if test_data is not None:
                self.model.eval()
                with torch.no_grad():
                    logits_ls, targets_ls = [], []
                    for batch in train_data:
                        batch = batch.to(device)
                        target = batch.s if pred_s else batch.y
                        logits_ls.append(self.forward(batch.x))
                        targets_ls.append(target)
                acc = accuracy(logits=torch.cat(logits_ls), targets=torch.cat(targets_ls))
                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()
