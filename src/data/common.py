from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Final, Generic, TypeVar, Union
from typing_extensions import TypeAliasType

from conduit.data import LoadedData, TernarySample, UnloadedData
from conduit.data.datasets import CdtDataset
from hydra.utils import to_absolute_path
from numpy import typing as npt
from torch import Tensor

__all__ = [
    "D",
    "Dataset",
    "DatasetFactory",
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
Dataset = TypeAliasType(
    "Dataset", CdtDataset[TernarySample[LoadedData], X, Tensor, Tensor], type_params=(X,)
)
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
    def __call__(self) -> Dataset[npt.NDArray]:
        raise NotImplementedError()
