from __future__ import annotations
from abc import abstractmethod
from typing_extensions import Self

from loguru import logger
import torch
import torch.nn as nn

from shared.configs.arguments import LoggingConf, MiscConf
from shared.data import DataModule

__all__ = ["Algorithm"]


class Algorithm(nn.Module):
    """Base class for algorithms."""

    def __init__(self, misc_cfg: MiscConf, *, log_cfg: LoggingConf) -> None:

        super().__init__()
        self.misc_cfg = misc_cfg
        self.log_cfg = log_cfg

        self.use_gpu = torch.cuda.is_available() and self.misc_cfg.gpu >= 0
        self.device = f"cuda:{self.misc_cfg.gpu}" if self.use_gpu else "cpu"
        self.use_amp = self.misc_cfg.use_amp and self.use_gpu
        logger.info(f"{torch.cuda.device_count()} GPUs available. Using device '{self.device}'")

    @abstractmethod
    def fit(self, dm: DataModule) -> Self:
        ...

    def run(self, dm: DataModule) -> Self:
        """Loads the data and fits and evaluates the model."""
        # Fit the model to the data
        self.fit(dm=dm)
        # finish logging for the current run
        run.finish()  # type: ignore
        return self
