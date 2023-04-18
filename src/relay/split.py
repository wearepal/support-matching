from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

from src.data import RandomSplitter
from src.data.splitter import RandomSplitter
from src.hydra_confs.camelyon17.conf import Camelyon17Conf
from src.hydra_confs.celeba.conf import CelebAConf
from src.hydra_confs.nih.conf import NIHChestXRayDatasetConf
from src.logging import WandbConf

__all__ = ["SplitRelay"]


@dataclass(eq=False)
class SplitRelay:
    ds: Any  # CdtDataset
    split: RandomSplitter = field(default_factory=RandomSplitter)
    wandb: WandbConf = field(default_factory=WandbConf)

    options: ClassVar[Dict[str, Dict[str, type]]] = {
        "ds": {
            "celeba": CelebAConf,
            "camelyon17": Camelyon17Conf,
            "nih": NIHChestXRayDatasetConf,
        }
    }

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        run = self.wandb.init(raw_config, suffix="artgen")
        self.split.save_as_artifact = True
        self.split(self.ds)
        if run is not None:
            run.finish()
