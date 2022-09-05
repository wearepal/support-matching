from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig

from src.algs.base import Algorithm
from src.models import Optimizer


@dataclass(eq=False)
class FsAlg(Algorithm):
    steps: int = 10_000
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_cls: Optimizer = Optimizer.ADAM
    optimizer_kwargs: Optional[DictConfig] = None
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    val_interval: float = 0.1
