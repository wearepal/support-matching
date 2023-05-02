from typing import Any, ClassVar, Dict, Optional

from attrs import define, field

from src.data import RandomSplitter
from src.data.splitter import RandomSplitter
from src.hydra_confs.datasets import Camelyon17Conf, CelebAConf, NIHChestXRayDatasetConf
from src.logging import WandbConf

__all__ = ["SplitRelay"]


@define(eq=False, kw_only=True)
class SplitRelay:
    defaults: list[Any] = field(default=[{"ds": "cmnist"}])

    ds: Any  # CdtDataset
    split: RandomSplitter = field(default=RandomSplitter)
    wandb: WandbConf = field(default=WandbConf)

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
