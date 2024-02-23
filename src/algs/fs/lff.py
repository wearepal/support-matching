from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar, Union
from typing_extensions import Self, override

from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import XI, LoadedData, SampleBase, SizedDataset, TernarySample, X
from conduit.types import Indexable, IndexType
from ranzen.misc import gcopy
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor
import torch.nn as nn

from src.data import DataModule, EvalTuple
from src.loss import GeneralizedCELoss
from src.models import Classifier

from .base import FsAlg

__all__ = ["IndexedDataset", "IndexedSample", "LabelEma", "LfF"]


class LabelEma(nn.Module, Indexable):
    labels: Tensor
    parameter: Tensor
    updated: Tensor

    def __init__(self, labels: Tensor, *, alpha: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer("labels", labels.flatten())
        self.register_buffer("parameter", torch.zeros(len(labels)))
        self.register_buffer("updated", torch.zeros(len(labels)))

    @torch.no_grad()  # pyright: ignore
    def update(self, data: Tensor, *, index: Union[Tensor, int]) -> None:
        self.parameter[index] = (
            self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    @torch.no_grad()  # pyright: ignore
    def max_loss(self, label: int) -> Tensor:
        label_index = self.labels == label
        return self.parameter[label_index].max()

    @override
    @torch.no_grad()  # pyright: ignore
    def __getitem__(self, index: IndexType) -> Tensor:
        return self.parameter[index].clone()


I = TypeVar("I", int, Tensor)


@dataclass(eq=False)
class IndexedSample(SampleBase[X]):
    s: Tensor
    y: Tensor
    idx: Tensor

    @override
    def __iter__(self) -> Iterator[Union[X, Tensor]]:
        yield from (self.x, self.y, self.s, self.idx)

    @override
    def __add__(self, other: Self) -> Self:
        copy = self._get_copy(other, is_batched=len(self.y) > 1)
        copy.s = torch.cat([copy.s, other.s], dim=0)
        copy.y = torch.cat([copy.y, other.y], dim=0)
        copy.idx = torch.cat([copy.idx, other.idx], dim=0)
        return copy

    @override
    def __getitem__(self: "IndexedSample[XI]", index: IndexType) -> "IndexedSample[XI]":
        return gcopy(
            self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index], idx=self.idx[index]
        )

    @classmethod
    def from_ts(cls, sample: TernarySample, *, idx: Tensor) -> Self:
        return cls(x=sample.x, y=sample.y, s=sample.s, idx=idx)


class IndexedDataset(SizedDataset[IndexedSample[Tensor]]):
    def __init__(self, dataset: CdtDataset[TernarySample[LoadedData], Any, Tensor, Tensor]) -> None:
        self.dataset = dataset

    @override
    def __getitem__(self, index: int) -> IndexedSample[Tensor]:
        sample = self.dataset[index]
        idx = torch.as_tensor(index, dtype=torch.long)
        return IndexedSample.from_ts(sample=sample, idx=idx)

    @override
    def __len__(self) -> int:
        return len(self.dataset)


@dataclass(kw_only=True, repr=False, eq=False, frozen=True)
class LfFClassifier(Classifier):
    criterion: CrossEntropyLoss
    sample_loss_ema_b: LabelEma
    sample_loss_ema_d: LabelEma
    q: float = 0.7

    @cached_property
    def biased_model(self) -> nn.Module:
        return gcopy(self.model, deep=True)

    @cached_property
    def biased_criterion(self) -> GeneralizedCELoss:
        return GeneralizedCELoss(q=self.q, reduction="mean")

    def training_step(self, batch: IndexedSample[Tensor], *, pred_s: bool = False) -> Tensor:  # type: ignore
        logit_b = self.biased_model(batch.x)
        logit_d = self.model(batch.x)
        loss_b = self.criterion.forward(logit_b, target=batch.y)
        with torch.no_grad():
            loss_d = self.criterion.forward(logit_d, target=batch.y)

        # EMA sample loss
        self.sample_loss_ema_b.update(loss_b.flatten(), index=batch.idx)
        self.sample_loss_ema_d.update(loss_d.flatten(), index=batch.idx)

        # class-wise normalize
        loss_b = self.sample_loss_ema_b[batch.idx]
        loss_d = self.sample_loss_ema_d[batch.idx]

        for c in range(logit_d.size(1)):
            class_index = batch.y == c
            max_loss_b = self.sample_loss_ema_b.max_loss(c)
            max_loss_d = self.sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        # re-weighting based on loss value / generalized CE for biased model
        eps = torch.finfo(loss_d.dtype).eps
        loss_weight = loss_b / (loss_b + loss_d + eps)
        loss_b_update = self.biased_criterion.forward(logit_b, target=batch.y)
        loss_d_update = self.criterion.forward(logit_d, target=batch.y, instance_weight=loss_weight)
        return loss_b_update + loss_d_update


@dataclass(repr=False, eq=False)
class LfF(FsAlg):
    alpha: float = 0.7
    q: float = 0.7

    @override
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        sample_loss_ema_b = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        sample_loss_ema_d = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        dm.train = IndexedDataset(dm.train)  # type: ignore
        classifier = LfFClassifier(
            criterion=CrossEntropyLoss(reduction="mean"),
            sample_loss_ema_b=sample_loss_ema_b,
            sample_loss_ema_d=sample_loss_ema_d,
            model=model,
            opt=self.opt,
            q=self.q,
        )
        classifier.fit(
            train_data=dm.train_dataloader(),
            test_data=dm.test_dataloader(),
            steps=self.steps,
            device=self.device,
            val_interval=self.val_interval,
            grad_scaler=self.grad_scaler,
            use_wandb=True,
            pred_s=False,
        )
        # Generate predictions with the trained model
        return classifier.predict(dm.test_dataloader(), device=self.device)
