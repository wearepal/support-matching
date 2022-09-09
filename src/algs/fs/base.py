from dataclasses import dataclass
from typing import Optional

from loguru import logger
from omegaconf import DictConfig
from ranzen import implements
from torch import Tensor
import torch.nn as nn

from src.algs.base import Algorithm
from src.data import DataModule, EvalTuple
from src.evaluation.metrics import EmEvalPair, SummaryMetric, compute_metrics
from src.models import Optimizer

__all__ = ["FsAlg"]


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
    monitor: SummaryMetric = SummaryMetric.ROB_ACC

    @property
    def alg_name(self) -> str:
        return self.__class__.__name__.lower()

    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        ...

    @implements(Algorithm)
    def run(self, dm: DataModule, *, model: nn.Module) -> Optional[float]:
        if dm.deployment_ids is not None:
            dm = dm.merge_train_and_deployment()
        et = self.routine(dm=dm, model=model)
        logger.info("Evaluating on the test set")
        pair = EmEvalPair.from_tensors(y_pred=et.y_pred, y_true=et.y_true, s=et.s, pred_s=False)
        alg_name = self.alg_name
        metrics = compute_metrics(
            pair=pair,
            model_name=alg_name,
            prefix="test",
            use_wandb=True,
            verbose=True,
        )
        return metrics.get(f"test/{self.monitor.value} ({alg_name})", None)
