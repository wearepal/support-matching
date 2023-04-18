from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from ranzen.hydra import Option
import torch.nn as nn

from src.algs.fs.base import FsAlg
from src.arch import BackboneFactory, PredictorFactory

from .base import BaseRelay

__all__ = ["FsRelay"]


@dataclass(eq=False)
class FsRelay(BaseRelay):
    alg: DictConfig = MISSING
    backbone: DictConfig = MISSING
    predictor: DictConfig = MISSING

    @classmethod
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        alg: List[Option],
        ds: List[Option],
        backbone: List[Option],
        predictor: List[Option],
        labeller: List[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            alg=alg,
            ds=ds,
            labeller=labeller,
            backbone=backbone,
            predictor=predictor,
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Any:
        run = self.init_wandb(raw_config, self.labeller, self.backbone, self.predictor)
        dm = self.init_dm()
        alg: FsAlg = instantiate(self.alg)
        backbone_fn: BackboneFactory = instantiate(self.backbone)
        predictor_fn: PredictorFactory = instantiate(self.predictor)
        backbone, out_dim = backbone_fn(input_dim=dm.dim_x[0])
        predictor, _ = predictor_fn(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)
        result = alg.run(dm=dm, model=model)
        if run is not None:
            run.finish()  # type: ignore
        return result
