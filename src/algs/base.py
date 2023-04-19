from abc import abstractmethod
from typing import Any, Iterator, Optional

import attrs
from loguru import logger
import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.data import DataModule, resolve_device

__all__ = ["Algorithm"]


@attrs.define(kw_only=True, repr=False, eq=False)
class Algorithm(nn.Module):
    """Base class for adversarial algorithms."""

    use_amp: bool = False  # Whether to use mixed-precision training
    gpu: int = 0  # which GPU to use (if available)
    max_grad_norm: Optional[float] = None

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.use_gpu: bool = torch.cuda.is_available() and self.gpu >= 0
        self.device: torch.device = resolve_device(self.gpu)
        self.use_amp = self.use_amp and self.use_gpu
        self.grad_scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None
        logger.info(f"{torch.cuda.device_count()} GPU(s) available - using device '{self.device}'")

    def _clip_gradients(self, parameters: Iterator[Parameter]) -> None:
        if (value := self.max_grad_norm) is not None:
            nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=value, norm_type=2.0)

    @abstractmethod
    def run(self, dm: DataModule, **kwargs: Any) -> Any:
        ...
