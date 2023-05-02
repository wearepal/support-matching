from typing import Any, ClassVar, Dict, Optional

from attrs import define, field

from src.data import RandomSplitter
from src.data.common import DatasetFactory
from src.data.nico_plus_plus import NICOPPCfg
from src.data.nih import NIHChestXRayDatasetCfg
from src.data.splitter import RandomSplitter
from src.hydra_confs.datasets import Camelyon17Cfg, CelebACfg
from src.logging import WandbConf

__all__ = ["SplitRelay"]


@define(eq=False, kw_only=True)
class SplitRelay:
    defaults: list[Any] = field(default=[{"ds": "cmnist"}, {"split": "random"}])

    ds: Any  # CdtDataset
    split: Any
    wandb: WandbConf = field(default=WandbConf)

    options: ClassVar[Dict[str, Dict[str, type]]] = {
        "ds": {
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
            "nicopp": NICOPPCfg,
        },
        "split": {"random": RandomSplitter},  # for compatibility we define a one-option variant
    }

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        assert isinstance(self.ds, DatasetFactory)
        assert isinstance(self.split, RandomSplitter)

        ds = self.ds()
        run = self.wandb.init(raw_config, (ds,), suffix="artgen")
        self.split.save_as_artifact = True
        self.split(ds)
        if run is not None:
            run.finish()
