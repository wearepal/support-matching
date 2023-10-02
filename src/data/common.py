from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Final, Generic, Protocol, TypeVar, Union
from typing_extensions import Self, TypeAlias

from conduit.data import LoadedData, SizedDataset, TernarySample, UnloadedData
from conduit.data.datasets import CdtDataset
from conduit.data.datasets.vision import CdtVisionDataset
from hydra.utils import to_absolute_path
import torch
from torch import Tensor

__all__ = [
    "D",
    "Dataset",
    "DatasetFactory",
    "PseudoCdtDataset",
    "TrainDepTestSplit",
    "find_data_dir",
    "process_data_dir",
]


DATA_DIRS: Final[dict[str, str]] = {
    "turing": "/srv/galene0/shared/data",
    "fear": "/srv/galene0/shared/data",
    "hydra": "/srv/galene0/shared/data",
    "goedel": "/srv/galene0/shared/data",
    "kyiv": "/srv/galene0/shared/data",
    "ada": "/srv/galene0/shared/data",
}


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return DATA_DIRS.get(name_of_machine, to_absolute_path("data"))


def process_data_dir(root: Union[Path, str, None]) -> Path:
    if root is None:
        return Path(find_data_dir())
    return Path(to_absolute_path(str(root))).resolve()


X = TypeVar("X", bound=UnloadedData)
Dataset: TypeAlias = CdtDataset[TernarySample[LoadedData], X, Tensor, Tensor]
D = TypeVar("D", bound=Dataset)


@dataclass
class TrainDepTestSplit(Generic[D]):
    train: D
    deployment: D
    test: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.deployment, self.test)

    def num_samples(self) -> int:
        return len(self.train) + len(self.deployment) + len(self.test)

    @property
    def num_samples_tr(self) -> int:
        return len(self.train)

    @property
    def num_samples_dep(self) -> int:
        return len(self.deployment)

    @property
    def num_samples_te(self) -> int:
        return len(self.test)


class DatasetFactory(ABC):
    @abstractmethod
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        raise NotImplementedError()


class PseudoCdtDataset(SizedDataset[TernarySample[LoadedData]], Protocol):
    """A protocol that captures all the behavior that we need from CdtDataset."""

    @property
    def s(self) -> Tensor:
        ...

    @s.setter
    def s(self, value: Tensor) -> None:
        ...

    @property
    def y(self) -> Tensor:
        ...

    @y.setter
    def y(self, value: Tensor) -> None:
        ...

    @property
    def dim_x(self) -> torch.Size:
        ...

    @property
    def dim_s(self) -> torch.Size:
        ...

    @property
    def dim_y(self) -> torch.Size:
        ...

    @property
    def card_y(self) -> int:
        ...

    @property
    def card_s(self) -> int:
        ...

    def __len__(self) -> int:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...
