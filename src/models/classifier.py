from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, Optional, Tuple, Union, overload
from typing_extensions import Literal

from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.metrics import hard_prediction
from conduit.types import Loss
from loguru import logger
from ranzen.torch.loss import CrossEntropyLoss
from ranzen.torch.utils import inf_generator
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm, trange
import wandb

from src.data import EvalTuple
from src.evaluation.metrics import EmEvalPair, compute_metrics

from .base import Model

__all__ = ["Classifier"]


@torch.no_grad()
def hard_prediction(logits: Tensor) -> Tensor:
    logits = torch.atleast_1d(logits.squeeze())
    return (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)


@torch.no_grad()
def soft_prediction(logits: Tensor) -> Tensor:
    logits = torch.atleast_1d(logits.squeeze())
    return logits.sigmoid() if logits.ndim == 1 else logits.softmax(dim=1)


@torch.no_grad()
def cat(*ls: list[Tensor], dim: int = 0) -> Iterator[Tensor]:
    for ls_ in ls:
        yield torch.cat(ls_, dim=dim)


@torch.no_grad()
def cat_cpu_flatten(*ls: list[Tensor], dim: int = 0) -> Iterator[Tensor]:
    for ls_ in ls:
        yield torch.cat(ls_, dim=dim).cpu().flatten()


@dataclass(eq=False)
class Classifier(Model):
    """Wrapper for classifier models equipped witht training/inference routines."""

    _PBAR_COL: ClassVar[str] = "#ffe252"
    criterion: Loss = field(default_factory=CrossEntropyLoss)

    @overload
    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: Literal[False] = ...,
    ) -> EvalTuple[None]:
        ...

    @overload
    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: Literal[True],
    ) -> EvalTuple[Tensor]:
        ...

    @torch.no_grad()
    def predict_dataset(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: bool = False,
    ) -> EvalTuple:
        self.to(device)
        hard_preds_ls, actual_ls, sens_ls, soft_preds_ls = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(data, desc="Generating predictions", colour=self._PBAR_COL):
                batch.to(device)
                logits = self.forward(batch.x)
                hard_preds_ls.append(hard_prediction(logits))
                actual_ls.append(batch.y)
                sens_ls.append(batch.s)
                if with_soft:
                    soft_preds_ls.append(soft_prediction(logits))

        hard_preds, actual, sens = cat_cpu_flatten(hard_preds_ls, actual_ls, sens_ls, dim=0)
        logger.info("Finished generating predictions")

        if with_soft:
            (soft_preds,) = cat_cpu_flatten(soft_preds_ls)
            return EvalTuple(y_pred=hard_preds, y_true=actual, s=sens, probs=soft_preds)
        return EvalTuple(y_pred=hard_preds, y_true=actual, s=sens)

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
        val_interval: int | float = 0.1,
        test_data: CdtDataLoader[TernarySample] | None = None,
        grad_scaler: Optional[GradScaler] = None,
        use_wandb: bool = False,
    ) -> None:
        use_amp = grad_scaler is not None
        # Test after every 20% of the total number of training iterations by default.
        if isinstance(val_interval, float):
            val_interval = max(1, round(val_interval * steps))
        self.to(device)
        self.train()

        pbar = trange(steps, desc="Training classifier", colour=self._PBAR_COL)
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

            if (test_data is not None) and (step > 0) and (step % val_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    preds_ls, targets_ls, groups_ls = [], [], []
                    for batch in tqdm(
                        test_data, desc="Validating classifier", colour=self._PBAR_COL
                    ):
                        batch = batch.to(device)
                        target = batch.s if pred_s else batch.y
                        with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                            logits = self.forward(batch.x)
                        preds_ls.append(hard_prediction(logits))
                        targets_ls.append(target)
                        groups_ls.append(batch.s)
                preds, targets, groups = cat_cpu_flatten(preds_ls, targets_ls, groups_ls, dim=0)
                pair = EmEvalPair.from_tensors(
                    y_pred=preds, y_true=targets, s=groups, pred_s=pred_s
                )
                metrics = compute_metrics(
                    pair=pair,
                    model_name=self.__class__.__name__.lower(),
                    use_wandb=use_wandb,
                    prefix="val",
                    verbose=False,
                )
                pbar.set_postfix(step=step + 1, **metrics)
            else:
                pbar.set_postfix(step=step + 1)
            pbar.update()

        pbar.close()
        logger.info("Finished training")
