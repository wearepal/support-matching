from collections.abc import Iterator
from dataclasses import dataclass
from typing import Union
from typing_extensions import Self, override

from conduit.data.structures import XI, LoadedData, TernarySample, X
from conduit.types import IndexType
from ranzen import gcopy
import torch
from torch import Tensor

from src.data.common import PseudoCdtDataset

__all__ = ["PriorSample", "PriorDataset"]


@dataclass
class PriorSample(TernarySample[X]):
    """A ternary sample that has a fourth element: a prior."""

    prior: Tensor

    @override
    def __iter__(self) -> Iterator[Union[X, Tensor]]:
        yield from (self.x, self.y, self.s, self.prior)

    @override
    def __add__(self, other: Self) -> Self:
        copy = super().__add__(other)
        copy.prior = torch.cat([copy.prior, other.prior], dim=0)
        return copy

    @override
    def __getitem__(self: "PriorSample[XI]", index: IndexType) -> "PriorSample[XI]":
        return gcopy(
            self,
            deep=False,
            x=self.x[index],
            y=self.y[index],
            s=self.s[index],
            idx=self.prior[index],
        )

    @classmethod
    def from_ts(cls, sample: TernarySample, *, prior: Tensor) -> Self:
        return cls(x=sample.x, y=sample.y, s=sample.s, prior=prior)


class PriorDataset(PseudoCdtDataset):
    """Wrapper around CdtDataset, which returns a PriorSample instead of TernarySample."""

    def __init__(self, dataset: PseudoCdtDataset, priors: Tensor) -> None:
        assert len(dataset) == priors.shape[0]
        self.dataset = dataset
        self.priors = priors

    @override
    def __getitem__(self, index: int) -> PriorSample[LoadedData]:
        sample = self.dataset[index]
        prior = self.priors[index]
        return PriorSample.from_ts(sample=sample, prior=prior)

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def s(self) -> Tensor:
        return self.dataset.s

    @s.setter
    def s(self, value: Tensor) -> None:
        self.dataset.s = value

    @property
    def y(self) -> Tensor:
        return self.dataset.y

    @y.setter
    def y(self, value: Tensor) -> None:
        self.dataset.y = value

    @property
    def dim_x(self) -> torch.Size:
        return self.dataset.dim_x

    @property
    def dim_s(self) -> torch.Size:
        return self.dataset.dim_s

    @property
    def dim_y(self) -> torch.Size:
        return self.dataset.dim_y

    @property
    def card_y(self) -> int:
        return self.dataset.card_y

    @property
    def card_s(self) -> int:
        return self.dataset.card_s

    def __add__(self, other: Self) -> Self:
        return PriorDataset(
            dataset=self.dataset + other.dataset,
            priors=torch.cat([self.priors, other.priors], dim=0),
        )

    def __iadd__(self, other: Self) -> Self:
        self.dataset += other.dataset
        self.priors = torch.cat([self.priors, other.priors], dim=0)
        return self
