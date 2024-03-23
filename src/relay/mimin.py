from dataclasses import dataclass, field
from typing import Any, ClassVar

from loguru import logger

from src.algs import MiMin
from src.algs.adv import Evaluator
from src.arch.autoencoder import AeFactory, AeFromArtifact, ResNetAE, SimpleConvAE, VqGanAe
from src.arch.predictors.fcn import Fcn
from src.data.common import DatasetFactory
from src.data.nih import NIHChestXRayDatasetCfg
from src.hydra_confs.datasets import Camelyon17Cfg, CelebACfg, ColoredMNISTCfg
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    Labeller,
    NullLabeller,
    UniformLabelNoiser,
)
from src.models import Model, OptimizerCfg, SplitAeCfg, SplitLatentAe

from .base import BaseRelay

__all__ = ["MiMinRelay"]


@dataclass(eq=False, kw_only=True)
class MiMinRelay(BaseRelay):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"ae_arch": "simple"},
            {"ds": "cmnist"},
            {"labeller": "none"},
        ]
    )

    alg: MiMin = field(default_factory=MiMin)
    ae_arch: AeFactory
    disc_arch: Fcn = field(default_factory=Fcn)
    disc: OptimizerCfg = field(default_factory=OptimizerCfg)
    eval: Evaluator = field(default_factory=Evaluator)
    ae: SplitAeCfg = field(default_factory=SplitAeCfg)
    ae_opt: OptimizerCfg = field(default_factory=OptimizerCfg)
    ds: DatasetFactory
    labeller: Labeller

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "cmnist": ColoredMNISTCfg,
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
        },
        "ae_arch": {
            "artifact": AeFromArtifact,
            "resnet": ResNetAE,
            "simple": SimpleConvAE,
            "vqgan": VqGanAe,
        },
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "gt": GroundTruthLabeller,
            "kmeans": KmeansOnClipEncodings,
            "artifact": LabelFromArtifact,
            "none": NullLabeller,
            "uniform_noise": UniformLabelNoiser,
        },
    }

    def run(self, raw_config: dict[str, Any] | None = None) -> None:
        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller, self.ae_arch, self.disc_arch))
        dm = self.init_dm(ds, self.labeller, device=self.alg.device)
        match dm.dim_x:
            case (c,):
                input_shape = (c, 0, 0)
            case (c, h, w):
                input_shape = (c, h, w)
            case _:
                raise ValueError(f"Unsupported input shape: {dm.dim_x}")
        ae_pair = self.ae_arch(input_shape=input_shape)
        ae = SplitLatentAe(
            opt=self.ae_opt,
            cfg=self.ae,
            model=ae_pair,
            feature_group_slices=dm.feature_group_slices,
        )
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        card_s = dm.card_s
        target_dim = card_s if card_s > 2 else 1
        disc_net, _ = self.disc_arch(input_dim=ae.encoding_size.zy, target_dim=target_dim)
        disc = Model(opt=self.disc, model=disc_net)
        self.alg.fit_and_evaluate(dm=dm, ae=ae, disc=disc, evaluator=self.eval)
        if run is not None:
            run.finish()
