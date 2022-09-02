from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Optional, Type, TypeVar, Union
from typing_extensions import Self

from conduit.data.datasets.base import CdtDataset
from conduit.data.structures import XI, LoadedData, SizedDataset, TernarySample, X
from conduit.types import Indexable, IndexType
from omegaconf import DictConfig
from ranzen import implements
from ranzen.misc import gcopy
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor
import torch.nn as nn

from src.data import DataModule
from src.evaluation.metrics import EvalPair, compute_metrics
from src.loss import GeneralizedCELoss
from src.models import Classifier, Optimizer

from .base import Algorithm

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

    @implements(Indexable)
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

    @implements(TernarySample)
    def __iter__(self) -> Iterator[LoadedData]:
        yield from (self.x, self.y, self.s, self.idx)

    @implements(TernarySample)
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.idx = torch.cat([copy.idx, other.idx], dim=0)
        return copy

    @implements(TernarySample)
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

    @implements(SizedDataset)
    def __getitem__(self, index: int) -> IndexedSample:
        sample = self.dataset[index]
        idx = torch.as_tensor(index, dtype=torch.long)
        return IndexedSample.from_ts(sample=sample, idx=idx)

    @implements(SizedDataset)
    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class _LabelEmaMixin:
    sample_loss_ema_b: LabelEma
    sample_loss_ema_d: LabelEma


@dataclass(eq=False)
class LfFClassifier(Classifier, _LabelEmaMixin):
    q: float = 0.7
    biased_model: nn.Module = field(init=False)
    biased_criterion: GeneralizedCELoss = field(init=False)
    criterion: CrossEntropyLoss = field(init=False)

    def __post_init__(self) -> None:
        self.biased_model = gcopy(self.model, deep=True)
        self.biased_criterion = GeneralizedCELoss(q=self.q, reduction="mean")
        self.criterion = CrossEntropyLoss(reduction="mean")
        super().__post_init__()

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


@dataclass(eq=False)
class LfF(Algorithm):
    steps: int = 10_000
    alpha: float = 0.7
    q: float = 0.7

    lr: float = 5.0e-4
    optimizer_cls: Optimizer = Optimizer.ADAM
    weight_decay: float = 0
    optimizer_kwargs: Optional[DictConfig] = None
    val_interval: float = 0.1

    @implements(Algorithm)
    def run(self, dm: DataModule, *, model: nn.Module) -> Self:
        if dm.deployment_ids is not None:
            dm = dm.merge_train_and_deployment()
        sample_loss_ema_b = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        sample_loss_ema_d = LabelEma(dm.train.y, alpha=self.alpha).to(self.device)
        dm.train = IndexedDataset(dm.train)
        classifier = LfFClassifier(
            model=model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            sample_loss_ema_b=sample_loss_ema_b,
            sample_loss_ema_d=sample_loss_ema_d,
            q=self.q,
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
        preds, labels, sens = classifier.predict_dataset(dm.test_dataloader(), device=self.device)
        pair = EvalPair.from_tensors(y_pred=preds, y_true=labels, s=sens, pred_s=False)
        compute_metrics(
            pair=pair,
            model_name=self.__class__.__name__.lower(),
            use_wandb=True,
            prefix="test",
        )
        return self
