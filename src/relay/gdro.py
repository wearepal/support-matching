from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
import torch.nn as nn

from src.algs.gdro import Gdro
from src.arch import BackboneFactory, PredictorFactory
from src.clustering.pipeline import ClusteringPipeline

from .base import BaseRelay

__all__ = ["GdroRelay"]


@dataclass(eq=False)
class GdroRelay(BaseRelay):
    backbone: DictConfig = MISSING
    predictor: DictConfig = MISSING
    clust: DictConfig = MISSING

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        dm = self.init_dm()
        run = self.init_wandb(raw_config, self.clust, self.backbone, self.predictor)

        # === Initialise the algorithm ===
        alg: Gdro = instantiate(self.alg)

        # === Cluster if not using the ground-truth labels for balancing ===
        if not dm.gt_deployment:
            # === Fit and evaluate the clusterer ===
            clusterer: ClusteringPipeline = instantiate(self.clust)()
            if hasattr(clusterer, "gpu"):
                # Set both phases to use the same device for convenience
                clusterer.gpu = alg.gpu  # type: ignore
            dm.deployment_ids = clusterer.run(dm=dm)

        backbone_fn: BackboneFactory = instantiate(self.backbone)
        predictor_fn: PredictorFactory = instantiate(self.predictor)
        backbone, out_dim = backbone_fn(input_dim=dm.dim_x[0])
        predictor, _ = predictor_fn(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)

        alg.run(dm=dm, model=model)

        run.finish()  # type: ignore
