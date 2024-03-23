from dataclasses import dataclass, field
import os
from typing import ClassVar

from loguru import logger
from ranzen.torch import random_seed
import torch

from src.data import DataModule, DataModuleConf, RandomSplitter, SplitFromArtifact
from src.data.common import Dataset
from src.data.splitter import DataSplitter, TabularSplitter
from src.labelling import Labeller
from src.logging import WandbConf

__all__ = ["BaseRelay"]


@dataclass(eq=False, kw_only=True)
class BaseRelay:
    dm: DataModuleConf = field(default_factory=DataModuleConf)
    split: DataSplitter
    wandb: WandbConf = field(default_factory=WandbConf)
    seed: int = 0

    options: ClassVar[dict[str, dict[str, type]]] = {
        "split": {
            "random": RandomSplitter,
            "artifact": SplitFromArtifact,
            "tabular": TabularSplitter,
        }
    }

    def init_dm(
        self,
        ds: Dataset,
        labeller: Labeller,
        device: torch.device,
    ) -> DataModule:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        dm = DataModule.from_ds(
            config=self.dm, ds=ds, splitter=self.split, labeller=labeller, device=device
        )
        logger.info(str(dm))
        return dm
