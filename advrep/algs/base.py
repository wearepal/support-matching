from __future__ import annotations
from abc import abstractmethod
import logging
from pathlib import Path
from typing_extensions import Self

import torch
import torch.nn as nn
import wandb
import yaml

from shared.configs.arguments import Config
from shared.data import DataModule
from shared.utils.utils import as_pretty_dict, flatten_dict, random_seed

__all__ = ["Algorithm"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Algorithm(nn.Module):
    """Base class for algorithms."""

    def __init__(
        self,
        cfg: Config,
    ) -> None:

        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.datamodule
        self.misc_cfg = cfg.train
        self.bias_cfg = cfg.split
        self.device = torch.device(self.misc_cfg.device)

    @abstractmethod
    def _fit(self, dm: DataModule) -> Self:
        ...

    def run(self) -> Self:
        """Loads the data and fits and evaluates the model."""

        random_seed(self.misc_cfg.seed, self.misc_cfg.use_gpu)

        group = f"{self.data_cfg.log_name}.{self.__class__.__name__}"
        if self.misc_cfg.log_method:
            group += "." + self.misc_cfg.log_method
        if self.misc_cfg.exp_group:
            group += "." + self.misc_cfg.exp_group
        if self.bias_cfg.log_dataset:
            group += "." + self.bias_cfg.log_dataset
        local_dir = Path(".", "local_logging")
        local_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity="predictive-analytics-lab",
            project="suds",
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(self.cfg)),
            group=group if group else None,
            reinit=True,
            mode=self.misc_cfg.wandb.name,
        )

        LOGGER.info(
            yaml.dump(
                as_pretty_dict(self.cfg),
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=True,
            )
        )

        # ==== construct dataset ====
        dm = DataModule.from_config(self.cfg)
        LOGGER.info(dm)
        # Fit the model to the data
        self._fit(dm=dm)
        # finish logging for the current run

        run.finish()  # type: ignore
        return self
