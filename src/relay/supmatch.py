from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements

from src.algs import SupportMatching
from src.algs.adv import Evaluator
from src.arch.autoencoder import AePair
from src.clustering.pipeline import ClusteringPipeline
from src.models import SplitLatentAe
from src.models.discriminator import NeuralDiscriminator

from .base import BaseRelay

__all__ = ["SupMatchRelay"]


@dataclass
class SupMatchRelay(BaseRelay):
    ae_arch: DictConfig = MISSING
    disc_arch: DictConfig = MISSING
    disc: DictConfig = MISSING
    clust: DictConfig = MISSING
    eval: DictConfig = MISSING
    ae: DictConfig = MISSING

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        dm = self.init_dm()
        run = self.init_wandb(raw_config, self.clust, self.ae_arch, self.disc_arch)

        # === Initialise the debiaser ===
        alg: SupportMatching = instantiate(self.alg)

        # === Cluster if not using the ground-truth labels for balancing ===
        if not dm.gt_deployment:
            # === Fit and evaluate the clusterer ===
            clusterer: ClusteringPipeline = instantiate(self.clust)()
            if hasattr(clusterer, "gpu"):
                # Set both phases to use the same device for convenience
                clusterer.gpu = alg.gpu  # type: ignore
            dm.deployment_ids = clusterer.run(dm=dm)

        # === Initialise the components of the debiaser ===
        ae_pair: AePair = instantiate(self.ae_arch)(input_shape=dm.dim_x)
        ae: SplitLatentAe = instantiate(self.ae, _partial_=True)(
            model=ae_pair,
            feature_group_slices=dm.feature_group_slices,
        )
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")

        disc_net, _ = instantiate(self.disc_arch)(
            input_dim=ae.encoding_size.zy,
            target_dim=1,
            batch_size=dm.batch_size_tr,
        )
        evaluator: Evaluator = instantiate(self.eval)
        disc: NeuralDiscriminator = instantiate(self.disc, model=disc_net)

        # === Train and evaluate the debiaser ===
        alg.run(dm=dm, ae=ae, disc=disc, evaluator=evaluator)

        run.finish()  # type: ignore
