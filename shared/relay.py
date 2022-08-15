from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ranzen.decorators import implements
from ranzen.hydra import Relay
from ranzen.torch import random_seed
import torch
import wandb

from advrep.algs.supmatch import SupportMatching
from shared.configs import Config
from shared.data import DataModule
from shared.utils.utils import as_pretty_dict, flatten_dict

__all__ = ["ASMRelay"]


@dataclass
class ASMRelay(Relay, Config):
    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        # ==== construct the data-module ====
        torch.multiprocessing.set_sharing_strategy("file_system")
        dm = DataModule.from_configs(
            dm_config=self.dm,
            ds_config=self.ds,
            split_config=self.split,
        )
        self.log(str(dm))
        # ==== set global variables ====
        random_seed(self.misc.seed, use_cuda=True)
        group = f"{dm.train.__class__.__name__}.{self.__class__.__name__}"
        if self.logging.log_method:
            group += "." + self.logging.log_method
        if self.logging.exp_group:
            group += "." + self.logging.exp_group
        if self.split:
            group += "." + self.split.log_dataset
        local_dir = Path(".", "local_logging")
        local_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity="predictive-analytics-lab",
            project="support-matching",
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(self)),
            group=group if group else None,
            reinit=True,
            mode=self.logging.mode.name,
        )

        # clusterer = Clusterer
        debiaser = SupportMatching(cfg=self)
        debiaser.run(dm=dm)
        run.finish()  # type: ignore
