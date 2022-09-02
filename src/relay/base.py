from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypeAlias

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay
from ranzen.torch import random_seed
import torch
import wandb

from src.data import DataModule, DataModuleConf, DataSplitter
from src.data.common import process_data_dir
from src.labelling import Labeller
from src.logging import WandbConf

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
    labeller: DictConfig = MISSING
    split: DictConfig = MISSING
    wandb: DictConfig = MISSING
    seed: int = 0

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
        **kwargs: List[Option],
    ) -> None:
        configs = dict(
            dm=[Option(DataModuleConf, name="base")],
            wandb=[Option(WandbConf, name="base")],
            split=[Option(DataSplitter, name="base")],
        )
        configs.update(kwargs)

        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    def init_dm(self) -> DataModule:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        splitter = instantiate(self.split)
        ds = instantiate(self.ds, root=process_data_dir(self.ds.root))
        # === Fit and evaluate the clusterer ===
        labeller: Labeller = instantiate(self.labeller)
        if hasattr(labeller, "gpu"):
            # Set both phases to use the same device for convenience
            labeller.gpu = alg.gpu  # type: ignore
        dm = DataModule.from_ds(
            config=self.dm,
            ds=ds,
            splitter=splitter,
            labeller=labeller,
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
