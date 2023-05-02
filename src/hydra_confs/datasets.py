from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from conduit.data.datasets.vision.camelyon17 import (
    Camelyon17Attr,
    Camelyon17Split,
    Camelyon17SplitScheme,
)
from conduit.data.datasets.vision.celeba import CelebASplit, CelebAttr
from omegaconf import MISSING

from src.data.nih import NiHSensAttr, NiHTargetAttr

__all__ = ["Camelyon17Conf", "CelebAConf", "ColoredMNISTConf", "NIHChestXRayDatasetConf"]


@dataclass
class Camelyon17Conf:
    _target_: str = "conduit.data.datasets.vision.camelyon17.Camelyon17"
    root: Union[str, Path] = MISSING
    download: bool = True
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[Union[Camelyon17Split, str]] = None
    split_scheme: Union[Camelyon17SplitScheme, str] = Camelyon17SplitScheme.OFFICIAL
    superclass: Union[Camelyon17Attr, str] = Camelyon17Attr.TUMOR
    subclass: Union[Camelyon17Attr, str] = Camelyon17Attr.CENTER


@dataclass
class CelebAConf:
    _target_: str = "conduit.data.datasets.vision.celeba.CelebA"
    root: Union[str, Path] = MISSING
    download: bool = True
    superclass: Union[CelebAttr, str] = CelebAttr.SMILING
    subclass: Union[CelebAttr, str] = CelebAttr.MALE
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[Union[CelebASplit, str]] = None


@dataclass
class ColoredMNISTConf:
    _target_: str = "conduit.data.datasets.vision.cmnist.ColoredMNIST"
    root: Union[str, Path] = MISSING
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


@dataclass
class NIHChestXRayDatasetConf:
    _target_: str = "src.data.nih.NIHChestXRayDataset"
    root: Union[Path, str] = MISSING
    sens_attr: NiHSensAttr = NiHSensAttr.gender
    target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.cardiomegaly
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
