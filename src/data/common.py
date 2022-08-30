from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Final, Generic, Iterator, TypeVar
from typing_extensions import TypeAlias

from conduit.data import CdtDataset, LoadedData, TernarySample
from hydra.utils import to_absolute_path
from torch import Tensor

__all__ = [
    "D",
    "Dataset",
    "TrainDepSplit",
    "find_data_dir",
    "process_data_dir",
]


DATA_DIRS: Final[dict[str, str]] = {
    "turing": "/srv/galene0/shared/data",
    "fear": "/srv/galene0/shared/data",
    "hydra": "/srv/galene0/shared/data",
    "goedel": "/srv/galene0/shared/data",
}


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return DATA_DIRS.get(name_of_machine, to_absolute_path("data"))


def process_data_dir(root: Path | str | None) -> Path:
    if root is None:
        return Path(find_data_dir())
    return Path(to_absolute_path(str(root))).resolve()


Dataset: TypeAlias = CdtDataset[TernarySample[LoadedData], Any, Tensor, Tensor]
D = TypeVar("D", bound=Dataset)


@dataclass
class TrainDepSplit(Generic[D]):
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
