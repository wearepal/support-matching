from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Union
from typing_extensions import TypeAlias

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Relay
from ranzen.torch import random_seed
import torch
import wandb

from src.algs import SupportMatching
from src.clustering.pipeline import ClusteringPipeline
from src.configs.classes import DataModuleConf
from src.data import DataModule
from src.data.common import process_data_dir

__all__ = ["SupMatchRelay"]


Run: TypeAlias = Union[
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


@dataclass
class SupMatchRelay(Relay):
    alg: DictConfig = MISSING
    ae_arch: DictConfig = MISSING
    disc_arch: DictConfig = MISSING
    disc: DictConfig = MISSING
    clust: DictConfig = MISSING
    dm: DataModuleConf = MISSING
    ds: DictConfig = MISSING
    eval: DictConfig = MISSING
    ae: DictConfig = MISSING
    seed: int = 0
    split: DictConfig = MISSING
    wandb: DictConfig = MISSING

    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        random_seed(self.seed, use_cuda=True)
        # ==== construct the data-module ====
        torch.multiprocessing.set_sharing_strategy("file_system")
        splitter = instantiate(self.split)
        ds = instantiate(self.ds, root=process_data_dir(self.ds.root))
        dm = DataModule.from_ds(
            config=self.dm,
            ds=ds,
            splitter=splitter,
        )
        logger.info(str(dm))
        if self.wandb.get("group", None) is None:
            default_group = f"{ds.__class__.__name__.lower()}_"
            default_group += "_".join(
                dict_conf["_target_"].split(".")[-1].lower()
                for dict_conf in (self.clust, self.ae_arch)
            )
            self.wandb["group"] = default_group
        run = instantiate(self.wandb, _partial_=True)(config=raw_config, reinit=True)
        # === Initialise the components ===
        encoder, decoder, latent_dim = instantiate(self.ae_arch)(input_shape=dm.dim_x)
        ae = instantiate(self.ae, _partial_=True)(
            encoder=encoder,
            decoder=decoder,
            feature_group_slices=dm.feature_group_slices,
        )
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")

        disc_net = instantiate(self.disc_arch)(
            input_dim=latent_dim,
            target_dim=1,
            batch_size=dm.batch_size_tr,
        )
        disc = instantiate(self.disc, model=disc_net)
        # === Initialise the debiaser ===
        alg: SupportMatching = instantiate(self.alg, _partial_=True)(
            ae=ae,
            disc=disc,
            evaluator=instantiate(self.eval),
        )
        # === Cluster if not using the ground-truth labels for balancing ===
        if not dm.gt_deployment:
            # === Fit and evaluate the clusterer ===
            logger.info("Initialising clustering")
            clusterer: ClusteringPipeline = instantiate(self.clust)()
            if hasattr(clusterer, "gpu"):
                # Set both phases to use the same device for convenience
                clusterer.gpu = alg.gpu  # type: ignore
            dm.deployment_ids = clusterer.run(dm=dm)
        # === Train and evaluate the debiaser ===
        alg.run(dm=dm)
        run.finish()  # type: ignore
