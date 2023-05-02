from typing import Any, ClassVar, Optional

from attrs import define, field
from conduit.data.datasets.vision import CdtVisionDataset
from loguru import logger

from src.algs import MiMin
from src.algs.adv import Evaluator
from src.arch.autoencoder import (
    AeFactory,
    AeFromArtifact,
    ResNetAE,
    SimpleConvAE,
    VqGanAe,
)
from src.arch.predictors.fcn import Fcn
from src.data.common import DatasetFactory
from src.data.nih import NIHChestXRayDatasetCfg
from src.hydra_confs.datasets import Camelyon17, CelebA, ColoredMNIST
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    Labeller,
    NullLabeller,
    UniformLabelNoiser,
)
from src.models import Model, ModelCfg, SplitLatentAe, SplitLatentAeCfg

from .base import BaseRelay

__all__ = ["MiMinRelay"]


@define(eq=False, kw_only=True)
class MiMinRelay(BaseRelay):
    defaults: list[Any] = field(
        default=[{"ae_arch": "simple"}, {"ds": "cmnist"}, {"labeller": "none"}]
    )

    alg: MiMin = field(default=MiMin)
    ae_arch: Any
    disc_arch: Fcn = field(default=Fcn)
    disc: ModelCfg = field(default=ModelCfg)
    eval: Evaluator = field(default=Evaluator)
    ae: SplitLatentAeCfg = field(default=SplitLatentAeCfg)
    ds: Any
    labeller: Any

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "cmnist": ColoredMNIST,
            "celeba": CelebA,
            "camelyon17": Camelyon17,
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

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> None:
        assert isinstance(self.ae_arch, AeFactory)
        assert isinstance(self.ds, CdtVisionDataset)
        assert isinstance(self.labeller, Labeller)
        assert isinstance(self.ds, DatasetFactory)

        run = self.wandb.init(raw_config, (self.ds, self.labeller, self.ae_arch, self.disc_arch))
        dm = self.init_dm(self.ds, self.labeller)
        ae_pair = self.ae_arch(input_shape=dm.dim_x)
        ae = SplitLatentAe(cfg=self.ae, model=ae_pair, feature_group_slices=dm.feature_group_slices)
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        card_s = dm.card_s
        target_dim = card_s if card_s > 2 else 1
        disc_net, _ = self.disc_arch(input_dim=ae.encoding_size.zy, target_dim=target_dim)
        disc = Model(cfg=self.disc, model=disc_net)
        self.alg.run(dm=dm, ae=ae, disc=disc, evaluator=self.eval)
        if run is not None:
            run.finish()
