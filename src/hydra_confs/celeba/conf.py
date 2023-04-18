from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from conduit.data.datasets.vision.celeba import CelebASplit, CelebAttr
from omegaconf import MISSING


@dataclass
class CelebAConf:
    _target_: str = "conduit.data.datasets.vision.celeba.CelebA"
    root: Union[str, Path] = MISSING
    download: bool = True
    superclass: Union[CelebAttr, str] = CelebAttr.SMILING
    subclass: Union[CelebAttr, str] = CelebAttr.MALE
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
    split: Optional[Union[CelebASplit, str]] = None
