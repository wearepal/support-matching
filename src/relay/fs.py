from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from typing_extensions import TypeAlias

from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
import torch.nn as nn

from src.algs import Erm, Gdro
from src.arch import BackboneFactory, PredictorFactory

from .base import BaseRelay

__all__ = ["FsRelay"]

Alg: TypeAlias = Union[Gdro, Erm]


@dataclass(eq=False)
class FsRelay(BaseRelay):
    backbone: DictConfig = MISSING
    predictor: DictConfig = MISSING

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        dm = self.init_dm()
        run = self.init_wandb(raw_config, self.labeller, self.backbone, self.predictor)
        # === Initialise the algorithm ===
        alg: Alg = instantiate(self.alg)
        backbone_fn: BackboneFactory = instantiate(self.backbone)
        predictor_fn: PredictorFactory = instantiate(self.predictor)
        backbone, out_dim = backbone_fn(input_dim=dm.dim_x[0])
        predictor, _ = predictor_fn(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)
        alg.run(dm=dm, model=model)

        run.finish()  # type: ignore
