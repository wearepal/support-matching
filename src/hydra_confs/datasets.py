from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union
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
    root: Union[str, Path]
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[Camelyon17Split] = None
    split_scheme: Camelyon17SplitScheme = Camelyon17SplitScheme.OFFICIAL
    superclass: Camelyon17Attr = Camelyon17Attr.TUMOR
    subclass: Camelyon17Attr = Camelyon17Attr.CENTER

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return Camelyon17(**asdict(self))


@dataclass
class CelebACfg(DatasetFactory):
    root: Union[str, Path]
    download: bool = True
    superclass: CelebAttr = CelebAttr.SMILING
    subclass: CelebAttr = CelebAttr.MALE
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[CelebASplit] = None

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return CelebA(**asdict(self))


@dataclass
class ColoredMNISTCfg(DatasetFactory):
    root: Union[str, Path]
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    label_map: Optional[dict[int, int]] = None
    colors: Optional[list[int]] = None
    num_colors: int = 10
    scale: float = 0.2
    correlation: Optional[float] = None
    binarize: bool = False
    greyscale: bool = False
    background: bool = False
    black: bool = True
    split: Any = None  # Optional[Union[ColoredMNISTSplit, str, List[int]]]
    seed: Optional[int] = 42

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return ColoredMNIST(**asdict(self))
