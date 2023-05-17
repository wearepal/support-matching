from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from torch import Tensor
import torch.nn as nn

from src.algs.base import Algorithm
from src.data import DataModule, EvalTuple
from src.evaluation.metrics import EmEvalPair, SummaryMetric, compute_metrics
from src.models import Optimizer

__all__ = ["FsAlg"]


@dataclass(repr=False, eq=False)
class FsAlg(Algorithm):
    steps: int = 10_000
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_cls: Optimizer = Optimizer.ADAM
    optimizer_kwargs: Optional[dict] = None
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None
    val_interval: float = 0.1
    monitor: SummaryMetric = SummaryMetric.ROB_ACC

    @property
    def alg_name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        raise NotImplementedError()

    def run(self, dm: DataModule, *, model: nn.Module) -> Optional[float]:
        if dm.deployment_ids is not None:
            dm = dm.merge_deployment_into_train()
        et = self.routine(dm=dm, model=model)
        logger.info("Evaluating on the test set")
        pair = EmEvalPair.from_tensors(y_pred=et.y_pred, y_true=et.y_true, s=et.s, pred_s=False)
        metrics = compute_metrics(pair=pair, prefix="test", use_wandb=True, verbose=True)
        return metrics.get(f"test/{self.monitor.value}", None)
