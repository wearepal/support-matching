from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, overload
from typing_extensions import Literal

from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.types import Loss
from loguru import logger
from ranzen.torch.loss import CrossEntropyLoss
from ranzen.torch.utils import inf_generator
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import trange
import wandb

from src.evaluation.metrics import EvalPair, compute_metrics

from .base import Model

__all__ = ["Classifier"]


@dataclass(eq=False)
class Classifier(Model):
    """Wrapper for classifier models equipped witht training/inference routines."""

    criterion: Loss = field(default_factory=CrossEntropyLoss)

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
        self.to(device)
        preds, actual, sens, soft_preds = [], [], [], []
        with torch.no_grad():
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

    def training_step(self, batch: TernarySample, *, pred_s: bool = False) -> Tensor:
        target = batch.s if pred_s else batch.y
        logits = self.forward(batch.x)
        return self.criterion(input=logits, target=target)

    def fit(
        self,
        train_data: CdtDataLoader[TernarySample],
        *,
        steps: int,
        device: torch.device,
        pred_s: bool = False,
        test_interval: int | float = 0.1,
        test_data: CdtDataLoader[TernarySample] | None = None,
        grad_scaler: Optional[GradScaler] = None,
        use_wandb: bool = False,
    ) -> None:
        use_amp = grad_scaler is not None
        logger.info("Training classifier")
        # Test after every 20% of the total number of training iterations by default.
        if isinstance(test_interval, float):
            test_interval = max(1, round(test_interval * steps))
        self.to(device)
        self.train()

        pbar = trange(steps)
        train_iter = inf_generator(train_data)
        for step in range(steps):
            batch = next(train_iter)
            batch = batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                loss = self.training_step(batch=batch)
                if use_wandb:
                    wandb.log({"train/loss": loss})

            if use_amp:  # Apply scaling for mixed-precision training
                loss = grad_scaler.scale(loss)  # type: ignore
            loss.backward()  # type: ignore
            self.step(grad_scaler=grad_scaler)
            self.optimizer.zero_grad()

            if (test_data is not None) and (step > 0) and (step % test_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    logits_ls, targets_ls, groups_ls = [], [], []
                    for batch in test_data:
                        batch = batch.to(device)
                        target = batch.s if pred_s else batch.y
                        with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                            logits_ls.append(self.forward(batch.x))
                        targets_ls.append(target)
                        groups_ls.append(batch.s)
                logits = torch.cat(logits_ls)
                targets = torch.cat(targets_ls)
                groups = torch.cat(groups_ls)

                pair = EvalPair.from_tensors(y_pred=logits, y_true=targets, s=groups, pred_s=pred_s)
                s_dim = len(groups.unique())
                metrics = compute_metrics(
                    pair=pair,
                    model_name=self.__class__.__name__.lower(),
                    step=0,
                    s_dim=s_dim,
                    use_wandb=use_wandb,
                    prefix="val",
                )
                pbar.set_postfix(step=step + 1, **metrics)
            else:
                pbar.set_postfix(step=step + 1)
            pbar.update()

        pbar.close()
        logger.info("Finished training")
