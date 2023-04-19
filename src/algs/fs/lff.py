from dataclasses import dataclass
from typing import Any, Generic, Iterator, Type, TypeVar, Union
from typing_extensions import Self, override

from attrs import define
from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import XI, LoadedData, SizedDataset, TernarySample, X
from conduit.types import Indexable, IndexType
from ranzen.misc import gcopy
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor
import torch.nn as nn

from src.data import DataModule, EvalTuple
from src.loss import GeneralizedCELoss
from src.models import Classifier
from src.models.base import ModelConf

from .base import FsAlg

__all__ = [
    "IndexedDataset",
    "IndexedSample",
    "LabelEma",
    "LfF",
]


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

    @torch.no_grad()
    def update(self, data: Tensor, *, index: Union[Tensor, int]) -> None:
        self.parameter[index] = (
            self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    @torch.no_grad()
    def max_loss(self, label: int) -> Tensor:
        label_index = self.labels == label
        return self.parameter[label_index].max()

    @override
    @torch.no_grad()
    def __getitem__(self, index: IndexType) -> Tensor:
        return self.parameter[index].clone()


I = TypeVar("I", int, Tensor)


@dataclass(eq=False)
class _IndexedSampleMixin(Generic[I]):
    idx: I


@dataclass(eq=False)
class IndexedSample(TernarySample[X], _IndexedSampleMixin[Tensor]):
    def add_field(self, *args: Any, **kwargs: Any) -> Self:
        return self

    @override
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s, self.idx)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.idx = torch.cat([copy.idx, other.idx], dim=0)
        return copy

    @override
    def __getitem__(self: "IndexedSample[XI]", index: IndexType) -> "IndexedSample[XI]":  # type: ignore
        return gcopy(
            self, deep=False, x=self.x[index], y=self.y[index], s=self.s[index], idx=self.idx[index]
        )

    @classmethod
    def from_ts(cls: Type[Self], sample: TernarySample, *, idx: Tensor) -> Self:
        return cls(x=sample.x, y=sample.y, s=sample.s, idx=idx)


class IndexedDataset(SizedDataset):
    def __init__(
        self,
        dataset: CdtDataset[TernarySample[LoadedData], Any, Tensor, Tensor],
    ) -> None:
        self.dataset = dataset

    @override
    def __getitem__(self, index: int) -> IndexedSample:
        sample = self.dataset[index]
        idx = torch.as_tensor(index, dtype=torch.long)
        return IndexedSample.from_ts(sample=sample, idx=idx)

    @override
    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class LfFClassifierConf(ModelConf):
    q: float = 0.7


class LfFClassifier(Classifier):
    biased_model: nn.Module
    biased_criterion: GeneralizedCELoss
    criterion: CrossEntropyLoss

    def __init__(
        self,
        cfg: LfFClassifierConf,
        model: nn.Module,
        sample_loss_ema_b: LabelEma,
        sample_loss_ema_d: LabelEma,
    ) -> None:
        super().__init__(cfg=cfg, model=model)
        self.sample_loss_ema_b = sample_loss_ema_b
        self.sample_loss_ema_d = sample_loss_ema_d
        self.biased_model = gcopy(self.model, deep=True)
        self.biased_criterion = GeneralizedCELoss(q=cfg.q, reduction="mean")
        self.criterion = CrossEntropyLoss(reduction="mean")

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


@define(kw_only=True, repr=False, eq=False)
class LfF(FsAlg):
    alpha: float = 0.7
    q: float = 0.7

    @override
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        sample_loss_ema_b = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        sample_loss_ema_d = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        dm.train = IndexedDataset(dm.train)
        classifier = LfFClassifier(
            model=model,
            cfg=LfFClassifierConf(
                lr=self.lr,
                weight_decay=self.weight_decay,
                optimizer_cls=self.optimizer_cls,
                optimizer_kwargs=self.optimizer_kwargs,
                scheduler_cls=self.scheduler_cls,
                scheduler_kwargs=self.scheduler_kwargs,
                q=self.q,
            ),
            sample_loss_ema_b=sample_loss_ema_b,
            sample_loss_ema_d=sample_loss_ema_d,
        )
        classifier.sample_loss_ema_b = sample_loss_ema_b
        classifier.sample_loss_ema_d = sample_loss_ema_d
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
