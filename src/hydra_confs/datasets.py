from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override

from conduit.data import TernarySample
import conduit.data.datasets.vision as cdt_vision
from conduit.data.datasets.vision import CdtVisionDataset, CelebASplit, CelebAttr
from conduit.data.datasets.vision.camelyon17 import (
    Camelyon17Attr,
    Camelyon17Split,
    Camelyon17SplitScheme,
)
from torch import Tensor

from src.data.common import DatasetFactory

__all__ = ["Camelyon17", "CelebA", "ColoredMNIST"]


@dataclass
class Camelyon17(DatasetFactory):
    root: Union[str, Path]
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[Camelyon17Split] = None
    split_scheme: Camelyon17SplitScheme = Camelyon17SplitScheme.OFFICIAL
    superclass: Camelyon17Attr = Camelyon17Attr.TUMOR
    subclass: Camelyon17Attr = Camelyon17Attr.CENTER

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return cdt_vision.Camelyon17(**asdict(self))


@dataclass
class CelebA(DatasetFactory):
    root: Union[str, Path]
    download: bool = True
    superclass: CelebAttr = CelebAttr.SMILING
    subclass: CelebAttr = CelebAttr.MALE
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[CelebASplit] = None

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return cdt_vision.CelebA(**asdict(self))


@dataclass
class ColoredMNIST(DatasetFactory):
    root: Union[str, Path]
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    label_map: Optional[Dict[str, int]] = None
    colors: Optional[List[int]] = None
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
        return cdt_vision.ColoredMNIST(**asdict(self))
