import os
from typing import Any, ClassVar

from attrs import define, field
from loguru import logger
from ranzen.torch import random_seed
import torch

from src.data import DataModule, DataModuleConf, RandomSplitter, SplitFromArtifact
from src.data.common import Dataset
from src.data.splitter import DataSplitter, TabularSplitter
from src.labelling import Labeller
from src.logging import WandbConf

__all__ = ["BaseRelay"]


@define(eq=False, kw_only=True)
class BaseRelay:
    dm: DataModuleConf = field(default=DataModuleConf)
    split: Any
    wandb: WandbConf = field(default=WandbConf)
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
        assert isinstance(self.split, DataSplitter)

        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        dm = DataModule.from_ds(
            config=self.dm, ds=ds, splitter=self.split, labeller=labeller, device=device
        )
        logger.info(str(dm))
        return dm
