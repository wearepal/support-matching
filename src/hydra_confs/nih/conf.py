from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import MISSING

from src.data.nih import NiHSensAttr, NiHTargetAttr


@dataclass
class NIHChestXRayDatasetConf:
    _target_: str = "src.data.nih.NIHChestXRayDataset"
    root: Union[Path, str] = MISSING
    sens_attr: NiHSensAttr = NiHSensAttr.gender
    target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.cardiomegaly
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
