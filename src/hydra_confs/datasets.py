from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing_extensions import override

from conduit.data import TernarySample
from conduit.data.datasets.vision import (
    Camelyon17,
    Camelyon17Split,
    Camelyon17SplitScheme,
    CdtVisionDataset,
    CelebA,
    CelebASplit,
    CelebAttr,
    ColoredMNIST,
)
from conduit.data.datasets.vision.camelyon17 import Camelyon17Attr
from torch import Tensor

from src.data.common import DatasetFactory

__all__ = ["Camelyon17Cfg", "CelebACfg", "ColoredMNISTCfg"]


@dataclass
class Camelyon17Cfg(DatasetFactory):
    root: str | Path
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Camelyon17Split | None = None
    split_scheme: Camelyon17SplitScheme = Camelyon17SplitScheme.OFFICIAL
    superclass: Camelyon17Attr = Camelyon17Attr.TUMOR
    subclass: Camelyon17Attr = Camelyon17Attr.CENTER

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return Camelyon17(**asdict(self))


@dataclass
class CelebACfg(DatasetFactory):
    root: str | Path
    download: bool = True
    superclass: CelebAttr = CelebAttr.SMILING
    subclass: CelebAttr = CelebAttr.MALE
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: CelebASplit | None = None

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return CelebA(**asdict(self))


@dataclass
class ColoredMNISTCfg(DatasetFactory):
    root: str | Path
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    label_map: dict[int, int] | None = None
    colors: list[int] | None = None
    num_colors: int = 10
    scale: float = 0.2
    correlation: float | None = None
    binarize: bool = False
    greyscale: bool = False
    background: bool = False
    black: bool = True
    split: Any = None  # Optional[Union[ColoredMNISTSplit, str, List[int]]]
    seed: int | None = 42

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return ColoredMNIST(**asdict(self))
