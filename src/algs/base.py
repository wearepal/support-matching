from __future__ import annotations
from torch.cuda.amp.grad_scaler import GradScaler
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from typing_extensions import Self

from loguru import logger
from ranzen.torch.module import DcModule
import torch

from src.data import DataModule, resolve_device

__all__ = ["Algorithm"]


@dataclass
class Algorithm(DcModule):
    """Base class for algorithms."""

    use_amp: bool = False  # Whether to use mixed-precision training
    gpu: int = 0  # which GPU to use (if available)
    use_gpu: bool = field(init=False)
    device: torch.device = field(init=False)
    grad_scaler: Optional[GradScaler] = field(init=False)

    def __post_init__(self) -> None:
        self.use_gpu = torch.cuda.is_available() and self.gpu >= 0
        self.device = resolve_device(self.gpu)
        self.use_amp = self.use_amp and self.use_gpu
        self.grad_scaler = GradScaler() if self.use_amp else None
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
