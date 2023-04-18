import os
from typing import ClassVar, Dict, Union
from typing_extensions import TypeAlias

from attrs import define, field
from conduit.data.datasets.vision import CdtVisionDataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ranzen.torch import random_seed
import torch
import wandb

from src.data import DataModule, DataModuleConf, RandomSplitter, SplitFromArtifact
from src.data.common import process_data_dir
from src.data.splitter import DataSplitter
from src.labelling import Labeller
from src.logging import WandbConf

__all__ = ["BaseRelay", "fix_dataset_root"]


Run: TypeAlias = Union[
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


@define(eq=False, kw_only=True)
class BaseRelay:
    dm: DataModuleConf = field(default=DataModuleConf)
    split: DataSplitter
    wandb: WandbConf = field(default=WandbConf)
    seed: int = 0

    options: ClassVar[Dict[str, Dict[str, type]]] = {
        "split": {"random": RandomSplitter, "artifact": SplitFromArtifact}
    }

    def init_dm(self, ds: CdtVisionDataset, labeller: Labeller) -> DataModule:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        splitter = self.split
        # ds = instantiate(self.ds, root=process_data_dir(self.ds.root))
        # labeller: Labeller = instantiate(self.labeller)
        dm = DataModule.from_ds(config=self.dm, ds=ds, splitter=splitter, labeller=labeller)
        logger.info(str(dm))
        return dm


def fix_dataset_root(hydra_config: DictConfig) -> None:
    """Deal with missing `root`."""

    if OmegaConf.is_missing(hydra_config["ds"], "root"):
        hydra_config["ds"]["root"] = process_data_dir(None)
    else:
        hydra_config["ds"]["root"] = process_data_dir(hydra_config["ds"]["root"])