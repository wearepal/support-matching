from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import MISSING


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
