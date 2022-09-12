from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from conduit.data.datasets.vision import Camelyon17, CelebA
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

from src.data import RandomSplitter
from src.data.common import process_data_dir
from src.data.nih import NIHChestXRayDataset
from src.data.splitter import RandomSplitter
from src.logging import WandbConf

from .base import BaseRelay

__all__ = ["ArtifactGenRelay"]


@dataclass(eq=False)
class ArtifactGenRelay(Relay):
    ds: DictConfig = MISSING
    split: DictConfig = MISSING
    wandb: DictConfig = MISSING

    @classmethod
    @implements(BaseRelay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            ds=[
                Option(CelebA, name="celeba"),
                Option(Camelyon17, name="camelyon17"),
                Option(NIHChestXRayDataset, name="nih"),
            ],
            split=[Option(RandomSplitter, name="random")],
            wandb=[Option(WandbConf, name="base")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        if self.wandb.get("group", None) is None:
            default_group = f"{self.ds['_target_'].lower()}_artgen"
            logger.info(f"No wandb group set - using {default_group} as the inferred default.")
            self.wandb["group"] = default_group
        run = instantiate(self.wandb, _partial_=True)(config=raw_config, reinit=True)
        splitter: RandomSplitter = instantiate(self.split, save_as_artifact=True)
        ds = instantiate(self.ds, root=process_data_dir(self.ds.root))
        splitter(ds)
        run.finish()  # type: ignore
