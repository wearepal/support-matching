from __future__ import annotations
from abc import abstractmethod
import logging
from typing_extensions import Self

import torch
from torch import Tensor
import torch.nn as nn

from shared.configs.arguments import Config
from shared.data import DataModule

__all__ = ["Algorithm"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Algorithm(nn.Module):
    """Base class for algorithms."""

    def __init__(self, cfg: Config) -> None:

        super().__init__()
        self.cfg = cfg
        self.misc_cfg = cfg.misc
        self.log_cfg = cfg.logging

        self.use_gpu = torch.cuda.is_available() and self.misc_cfg.gpu >= 0
        self.device = f"cuda:{self.misc_cfg.gpu}" if self.use_gpu else "cpu"
        self.use_amp = self.misc_cfg.use_amp and self.use_gpu
        LOGGER.info(f"{torch.cuda.device_count()} GPUs available. Using device '{self.device}'")

    @abstractmethod
    def fit(self, dm: DataModule, *, group_ids: Tensor | None = None) -> Self:
        ...

    def run(self, dm: DataModule) -> Self:
        """Loads the data and fits and evaluates the model."""
        # Fit the model to the data
        self.fit(dm=dm)
        # finish logging for the current run
        run.finish()  # type: ignore
        return self
