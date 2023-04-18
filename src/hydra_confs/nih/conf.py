from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from src.data.nih import NiHSensAttr
from src.data.nih import NiHTargetAttr
from typing import Any
from typing import Optional
from typing import Union


@dataclass
class NIHChestXRayDatasetConf:
    _target_: str = "src.data.nih.NIHChestXRayDataset"
    root: Union[Path, str] = MISSING
    sens_attr: NiHSensAttr = NiHSensAttr.gender
    target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.cardiomegaly
    transform: Any = None  # Optional[Union[Compose, BasicTransform, Callable[[Image], Any]]]
