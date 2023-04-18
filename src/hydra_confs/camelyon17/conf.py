from dataclasses import dataclass, field
from conduit.data.datasets.vision.camelyon17 import Camelyon17Attr
from conduit.data.datasets.vision.camelyon17 import Camelyon17Split
from conduit.data.datasets.vision.camelyon17 import Camelyon17SplitScheme
from omegaconf import MISSING
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union


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
