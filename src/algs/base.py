from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from loguru import logger
from ranzen.torch.module import DcModule
import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.data import DataModule, resolve_device

__all__ = ["Algorithm"]


@dataclass(repr=False, eq=False)
class Algorithm(DcModule):
    """Base class for adversarial algorithms."""

    use_amp: bool = False  # Whether to use mixed-precision training
    gpu: int = 0  # which GPU to use (if available)
    max_grad_norm: Optional[float] = None

    use_gpu: bool = field(init=False)
    device: torch.device = field(init=False)
    grad_scaler: Optional[GradScaler] = field(init=False)

    def __post_init__(self) -> None:
        self.use_gpu = torch.cuda.is_available() and self.gpu >= 0
        self.device = resolve_device(self.gpu)
        self.use_amp = self.use_amp and self.use_gpu
        self.grad_scaler = GradScaler() if self.use_amp else None
        logger.info(f"{torch.cuda.device_count()} GPU(s) available - using device '{self.device}'")

    def _clip_gradients(self, parameters: Iterator[Parameter]) -> None:
        if (value := self.max_grad_norm) is not None:
            nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=value, norm_type=2.0)

    @abstractmethod
    def run(self, dm: DataModule, **kwargs: Any) -> Any:
        raise NotImplementedError()
