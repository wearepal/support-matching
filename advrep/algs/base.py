from __future__ import annotations
from abc import abstractmethod
import logging
from pathlib import Path
from typing_extensions import Self

from ranzen.torch import random_seed
import torch
from torch import Tensor
import torch.nn as nn
import wandb

from shared.configs.arguments import Config
from shared.data import DataModule
from shared.utils.utils import as_pretty_dict, flatten_dict

__all__ = ["Algorithm"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Algorithm(nn.Module):
    """Base class for algorithms."""

    def __init__(self, cfg: Config) -> None:

        super().__init__()
        self.cfg = cfg
        self.misc_cfg = cfg.misc
        self.ds_cfg = cfg.ds
        self.dm_cfg = cfg.dm
        self.split_cfg = cfg.split
        self.log_cfg = cfg.logging

        self.use_gpu = torch.cuda.is_available() and self.misc_cfg.gpu >= 0
        self.device = f"cuda:{self.misc_cfg.gpu}" if self.use_gpu else "cpu"
        self.use_amp = self.misc_cfg.use_amp and self.use_gpu
        LOGGER.info(f"{torch.cuda.device_count()} GPUs available. Using device '{self.device}'")

    @abstractmethod
    def fit(self, dm: DataModule, *, group_ids: Tensor | None = None) -> Self:
        ...

    def run(self) -> Self:
        """Loads the data and fits and evaluates the model."""

        # ==== set global variables ====
        random_seed(self.misc_cfg.seed, use_cuda=self.use_gpu)
        torch.multiprocessing.set_sharing_strategy("file_system")

        # ==== construct the data-module ====
        dm = DataModule.from_configs(
            dm_config=self.cfg.dm,
            ds_config=self.cfg.ds,
            split_config=self.cfg.split,
        )
        LOGGER.info(dm)

        group = f"{dm.train.__class__.__name__}.{self.__class__.__name__}"
        if self.log_cfg.log_method:
            group += "." + self.log_cfg.log_method
        if self.log_cfg.exp_group:
            group += "." + self.log_cfg.exp_group
        if self.split_cfg.log_dataset:
            group += "." + self.split_cfg.log_dataset
        local_dir = Path(".", "local_logging")
        local_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity="predictive-analytics-lab",
            project="support-matching",
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(self.cfg)),
            group=group if group else None,
            reinit=True,
            mode=self.log_cfg.mode.name,
        )

        # Fit the model to the data
        self.fit(dm=dm)
        # finish logging for the current run
        run.finish()  # type: ignore
        return self
