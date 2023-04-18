from typing import Optional, Union

import attrs
from conduit.types import Loss
from ranzen import gcopy, implements
from ranzen.torch import WeightedBatchSampler
from torch import Tensor
import torch.nn as nn

from src.data import DataModule, EvalTuple
from src.models import Classifier
from src.models.base import ModelConf

from .base import FsAlg

__all__ = ["Jtt"]


@attrs.define(kw_only=True, repr=False, eq=False)
class Jtt(FsAlg):
    id_steps: Union[float, int] = 0.02
    lambda_uw: Optional[float] = None
    criterion: Optional[Loss] = None

    def __attrs_post_init__(self) -> None:
        if isinstance(self.id_steps, float):
            if not (0 <= self.id_steps <= 1):
                raise AttributeError("If 'id_steps' is a float, it must be in the range [0, 1].")

    @implements(FsAlg)
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        model_id = gcopy(model, deep=True)
        # Stage one: identification
        classifier = Classifier(
            model=model_id,
            cfg=ModelConf(
                lr=self.lr,
                weight_decay=self.weight_decay,
                optimizer_cls=self.optimizer_cls,
                optimizer_kwargs=self.optimizer_kwargs,
                scheduler_cls=self.scheduler_cls,
                scheduler_kwargs=self.scheduler_kwargs,
            ),
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
        et = classifier.predict(dm.train_dataloader(eval=True), device=self.device)
        del model_id
        # Stage two: upweighting identified points
        correct = et.y_pred.flatten() == et.y_true.flatten()
        error_set = (~correct).nonzero().squeeze(-1)
        weights = correct.float()
        lambda_uw = len(dm.train) / len(error_set) if self.lambda_uw is None else self.lambda_uw
        weights.index_fill_(dim=0, index=error_set, value=lambda_uw)
        batch_sampler = WeightedBatchSampler(
            weights=weights, batch_size=dm.cfg.batch_size_tr, replacement=True
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
        return classifier.predict(dm.test_dataloader(), device=self.device)
