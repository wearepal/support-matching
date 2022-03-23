from __future__ import annotations
from abc import abstractmethod
import logging
from pathlib import Path
from typing_extensions import Self

from ranzen.torch import random_seed
import torch
import torch.nn as nn
import wandb
import yaml

from shared.configs.arguments import Config
from shared.data import DataModule
from shared.utils.loadsave import ClusterResults
from shared.utils.utils import as_pretty_dict, flatten_dict

__all__ = ["Algorithm"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Algorithm(nn.Module):
    """Base class for algorithms."""

    def __init__(self, cfg: Config) -> None:

        super().__init__()
        self.cfg = cfg
        self.train_cfg = cfg.misc
        self.ds_cfg = cfg.ds
        self.dm_cfg = cfg.dm
        self.split_cfg = cfg.split
        self.log_cfg = cfg.logging
        self.device = torch.device(self.train_cfg.device)

    @abstractmethod
    def fit(self, dm: DataModule, cluster_results: ClusterResults | None = None) -> Self:
        ...

    def run(self) -> Self:
        """Loads the data and fits and evaluates the model."""

        random_seed(self.train_cfg.seed, use_cuda=self.train_cfg.use_gpu)

        ds_name = self.ds_cfg["_target_"]
        group = f"{ds_name}.{self.__class__.__name__}"
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

        LOGGER.info(
            yaml.dump(
                as_pretty_dict(self.cfg),
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=True,
            )
        )

        # ==== construct the data-module ====
        dm = DataModule.from_configs(
            dm_config=self.cfg.dm,
            ds_config=self.cfg.ds,
            split_config=self.cfg.split,
        )
        LOGGER.info(dm)
        # Fit the model to the data
        self.fit(dm=dm)
        # finish logging for the current run
        run.finish()  # type: ignore
        return self
