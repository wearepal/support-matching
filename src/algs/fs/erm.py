from dataclasses import dataclass, field
from typing import Optional

from conduit.types import Loss
from ranzen import implements
from ranzen.torch import CrossEntropyLoss, ReductionType
from torch import Tensor
import torch.nn as nn

from src.data import DataModule
from src.data.utils import EvalTuple
from src.models import Classifier

from .base import FsAlg

__all__ = ["Erm"]


@dataclass(eq=False)
class Erm(FsAlg):
    criterion: Optional[Loss] = None

    @implements(FsAlg)
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        classifier = Classifier(
            model=model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            criterion=self.criterion,
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
        return classifier.predict(dm.test_dataloader(), device=self.device)
