from dataclasses import dataclass
from typing import Any
from typing_extensions import override

from ranzen import gcopy
from ranzen.torch import WeightedBatchSampler
from torch import Tensor
import torch.nn as nn

from src.data import DataModule, EvalTuple
from src.models import Classifier

from .base import FsAlg

__all__ = ["Jtt"]


@dataclass(repr=False, eq=False)
class Jtt(FsAlg):
    id_steps: float | int = 0.02
    lambda_uw: float | None = None
    criterion: Any = None  # Optional[Loss]

    def __post_init__(self) -> None:
        if isinstance(self.id_steps, float):
            if not (0 <= self.id_steps <= 1):
                raise AttributeError("If 'id_steps' is a float, it must be in the range [0, 1].")
        super().__post_init__()

    @override
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        model_id = gcopy(model, deep=True)
        # Stage one: identification
        classifier = Classifier(model=model_id, opt=self.opt, criterion=self.criterion)
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
        return classifier.predict(dm.test_dataloader(), device=self.device)
