from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.data import DataSplitter, RandomSplitter
from src.data.common import DatasetFactory
from src.data.factories import NICOPPCfg
from src.data.nih import NIHChestXRayDatasetCfg
from src.hydra_confs.datasets import Camelyon17Cfg, CelebACfg
from src.logging import WandbConf

__all__ = ["SplitRelay"]


@dataclass(eq=False, kw_only=True)
class SplitRelay:
    defaults: list[Any] = field(default_factory=lambda: [{"ds": "celeba"}, {"split": "random"}])

    ds: DatasetFactory
    split: DataSplitter
    wandb: WandbConf = field(default_factory=WandbConf)

    options: ClassVar[dict[str, dict[str, type]]] = {
        "ds": {
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
            "nicopp": NICOPPCfg,
        },
        "split": {"random": RandomSplitter},  # for compatibility we define a one-option variant
    }

    def run(self, raw_config: dict[str, Any] | None = None) -> None:
        assert isinstance(self.split, RandomSplitter)

        ds = self.ds()
        run = self.wandb.init(raw_config, (ds,), with_tag="artgen")
        self.split.save_as_artifact = True
        self.split(ds)
        if run is not None:
            run.finish()
