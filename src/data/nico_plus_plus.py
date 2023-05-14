"""NICO Dataset."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from typing_extensions import override

from conduit.data.datasets.vision import CdtVisionDataset, NICOPP, NicoPPTarget
from conduit.data.structures import TernarySample
from torch import Tensor

from src.data.common import DatasetFactory

__all__ = ["NICOPPCfg"]


@dataclass
class NICOPPCfg(DatasetFactory):
    root: Union[Path, str]
    target_attrs: Optional[list[NicoPPTarget]] = None
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]

    @override
    def __call__(self) -> CdtVisionDataset[TernarySample, Tensor, Tensor]:
        return NICOPP(root=self.root, transform=self.transform, superclasses=self.target_attrs)
