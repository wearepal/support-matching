from __future__ import annotations

import logging
import os
from abc import abstractmethod
from pathlib import Path

import torch.nn as nn
import wandb
import yaml
from torch.tensor import Tensor

from shared.configs.arguments import Config
from shared.data.data_loading import DatasetTriplet, load_dataset
from shared.utils.utils import as_pretty_dict, flatten_dict, random_seed

__all__ = ["AlgBase"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class AlgBase(nn.Module):
    """Base class for algorithms."""

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.misc_cfg = cfg.misc
        self.bias_cfg = cfg.bias

    def _to_device(self, *tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Place tensors on the correct device."""
        moved = [tensor.to(self.misc_cfg.device, non_blocking=True) for tensor in tensors]

        return moved[0] if len(moved) == 1 else tuple(moved)

    @abstractmethod
    def _fit(self, datasets: DatasetTriplet) -> AlgBase:
        ...

    def run(self) -> None:
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
        datasets: DatasetTriplet = load_dataset(self.cfg)
        LOGGER.info(
            "Size of context-set: {}, training-set: {}, test-set: {}".format(
                len(datasets.context),  # type: ignore
                len(datasets.train),  # type: ignore
                len(datasets.test),  # type: ignore
            )
        )
        # Fit the model to the data
        self._fit(datasets=datasets)
        # finish logging for the current run
        run.finish()
