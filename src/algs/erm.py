from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from typing_extensions import Self

from loguru import logger
from omegaconf import DictConfig
from ranzen import implements
import torch.nn as nn

from src.data import DataModule
from src.evaluation.metrics import EvalPair, compute_metrics
from src.models import Classifier, Optimizer

from .base import Algorithm

__all__ = ["Erm"]


@dataclass(eq=False)
class Erm(Algorithm):
    steps: int = 10_000
    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 5.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[DictConfig] = None
    val_interval: float = 0.1

    @implements(Algorithm)
    def run(self, dm: DataModule, *, model: nn.Module) -> Self:
        if dm.deployment_ids is not None:
            dm = dm.merge_train_and_deployment()
        classifier = Classifier(
            model=model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        classifier.fit(
            train_data=dm.train_dataloader(),
            test_data=dm.test_dataloader(),
            steps=self.steps,
            val_interval=self.val_interval,
            device=self.device,
            grad_scaler=self.grad_scaler,
            use_wandb=True,
        )

        # Generate predictions with the trained model
        preds, labels, sens = classifier.predict_dataset(dm.test_dataloader(), device=self.device)
        logger.info("Evaluating on the test set")
        pair = EvalPair.from_tensors(y_pred=preds, y_true=labels, s=sens, pred_s=False)
        compute_metrics(
            pair=pair,
            model_name=self.__class__.__name__.lower(),
            use_wandb=True,
        )
        return self
