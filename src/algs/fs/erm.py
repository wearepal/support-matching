from dataclasses import dataclass
from typing import Any
from typing_extensions import override

from torch import Tensor
import torch.nn as nn

from src.data import DataModule
from src.data.utils import EvalTuple
from src.models import Classifier

from .base import FsAlg

__all__ = ["Erm"]


@dataclass(repr=False, eq=False)
class Erm(FsAlg):
    criterion: Any = None  # Optional[Loss]

    @override
    def routine(self, dm: DataModule, *, model: nn.Module) -> EvalTuple[Tensor, None]:
        classifier = Classifier(model=model, opt=self.opt, criterion=self.criterion)
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
