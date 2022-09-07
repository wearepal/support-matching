from dataclasses import dataclass, field
from typing import Optional, Union
from typing_extensions import Self

from conduit.types import Loss
from loguru import logger
from ranzen import gcopy, implements
from ranzen.torch import CrossEntropyLoss, WeightedBatchSampler
import torch.nn as nn

from src.data import DataModule
from src.evaluation.metrics import EvalPair, compute_metrics
from src.models import Classifier

from .base import FsAlg

__all__ = ["Jtt"]


@dataclass(eq=False)
class Jtt(FsAlg):
    criterion: Loss = field(default_factory=CrossEntropyLoss)
    id_steps: Union[float, int] = 0.02
    lambda_uw: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.id_steps, float):
            if not (0 <= self.id_steps <= 1):
                raise AttributeError("'id_steps' must be in the range [0, 1].")

    @implements(FsAlg)
    def run(self, dm: DataModule, *, model: nn.Module) -> Self:
        if dm.deployment_ids is not None:
            dm = dm.merge_train_and_deployment()
        model_id = gcopy(model, deep=True)
        # Stage one: identification
        classifier = Classifier(
            model=model_id,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            criterion=self.criterion,
        )
        id_steps = (
            self.id_steps if isinstance(self.id_steps, int) else round(self.id_steps * self.steps)
        )
        classifier.fit(
            train_data=dm.train_dataloader(),
            test_data=dm.test_dataloader(),
            steps=id_steps,
            val_interval=self.val_interval,
            device=self.device,
            grad_scaler=self.grad_scaler,
            use_wandb=True,
        )
        # Generate predictions with the trained model
        preds, labels, _ = classifier.predict_dataset(
            dm.train_dataloader(eval=True), device=self.device
        )
        del model_id
        # Stage two: upweighting identified points
        correct = preds.flatten() == labels.flatten()
        error_set = (~correct).nonzero().squeeze(-1)
        weights = correct.float()
        lambda_uw = len(dm.train) / len(error_set) if self.lambda_uw is None else self.lambda_uw
        weights.index_fill_(dim=0, index=error_set, value=lambda_uw)
        batch_sampler = WeightedBatchSampler(
            weights=weights, batch_size=dm.batch_size_tr, replacement=True
        )
        classifier.fit(
            train_data=dm.train_dataloader(batch_sampler=batch_sampler),
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
            prefix="test",
            use_wandb=True,
            verbose=True,
        )
        return self
