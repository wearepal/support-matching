from dataclasses import dataclass
from typing import Iterator, Optional

from loguru import logger
from ranzen.torch.module import DcModule
import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.data import resolve_device

__all__ = ["Algorithm"]


@dataclass(repr=False, eq=False)
class Algorithm(DcModule):
    """Base class for adversarial algorithms."""

    use_amp: bool = False  # Whether to use mixed-precision training
    gpu: int = 0  # which GPU to use (if available)
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        self.use_gpu: bool = torch.cuda.is_available() and self.gpu >= 0
        self.device: torch.device = resolve_device(self.gpu)
        self.use_amp = self.use_amp and self.use_gpu
        self.grad_scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None
        logger.info(f"{torch.cuda.device_count()} GPU(s) available - using device '{self.device}'")

    def _clip_gradients(self, parameters: Iterator[Parameter]) -> None:
        if (value := self.max_grad_norm) is not None:
            nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=value, norm_type=2.0)
