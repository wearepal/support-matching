from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Union
from typing_extensions import TypeAlias

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.hydra import Relay
from ranzen.torch import random_seed
import torch
import wandb

from src.data.common import process_data_dir
from src.data.data_module import DataModule, DataModuleConf

__all__ = ["BaseRelay"]


Run: TypeAlias = Union[
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


@dataclass(eq=False)
class BaseRelay(Relay):
    alg: DictConfig = MISSING
    dm: DataModuleConf = MISSING
    ds: DictConfig = MISSING
    split: DictConfig = MISSING
    wandb: DictConfig = MISSING
    seed: int = 0

    def init_dm(self) -> DataModule:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        splitter = instantiate(self.split)
        ds = instantiate(self.ds, root=process_data_dir(self.ds.root))
        dm = DataModule.from_ds(
            config=self.dm,
            ds=ds,
            splitter=splitter,
        )
        logger.info(str(dm))
        return dm

    def init_wandb(self, raw_config: Optional[Dict[str, Any]] = None, *confs: DictConfig) -> Run:
        if self.wandb.get("group", None) is None:
            default_group = f"{self.ds['_target_'].lower()}_"
            default_group += "_".join(
                dict_conf["_target_"].split(".")[-1].lower() for dict_conf in confs
            )
            self.wandb["group"] = default_group
        return instantiate(self.wandb, _partial_=True)(config=raw_config, reinit=True)
