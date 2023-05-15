from dataclasses import asdict
from typing import Any, ClassVar, Dict, Optional

from attrs import define, field
from loguru import logger

from src.algs import SupportMatching
from src.algs.adv import Evaluator, NeuralScorer, NullScorer, Scorer
from src.arch.autoencoder import (
    AeFactory,
    AeFromArtifact,
    ResNetAE,
    SimpleConvAE,
    VqGanAe,
    save_ae_artifact,
)
from src.arch.predictors.base import PredictorFactory
from src.arch.predictors.fcn import Fcn, SetFcn
from src.data.common import DatasetFactory
from src.data.nico_plus_plus import NICOPPCfg
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
from src.models import SplitLatentAe
from src.models.autoencoder import SplitLatentAeCfg
from src.models.discriminator import NeuralDiscriminator, NeuralDiscriminatorCfg
from src.utils import full_class_path

from .base import BaseRelay

__all__ = ["SupMatchRelay"]


@define(eq=False, kw_only=True)
class SupMatchRelay(BaseRelay):
    defaults: list[Any] = field(
        default=[
            {"ae_arch": "simple"},
            {"ds": "cmnist"},
            {"disc_arch": "set"},
            {"labeller": "none"},
            {"scorer": "none"},
            {"split": "random"},
        ]
    )
    alg: SupportMatching = field(default=SupportMatching)
    ae: SplitLatentAeCfg = field(default=SplitLatentAeCfg)
    ae_arch: Any  # AeFactory
    ds: Any  # DatasetFactory
    disc_arch: Any  # PredictorFactory
    disc: NeuralDiscriminatorCfg = field(default=NeuralDiscriminatorCfg)
    eval: Evaluator = field(default=Evaluator)
    labeller: Any  # Labeller
    scorer: Any  # Scorer
    artifact_name: Optional[str] = None
    """Save model weights under this name."""

    options: ClassVar[Dict[str, Dict[str, type]]] = BaseRelay.options | {
        "scorer": {"neural": NeuralScorer, "none": NullScorer},
        "ds": {
            "cmnist": ColoredMNISTCfg,
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
            "nicopp": NICOPPCfg,
        },
        "ae_arch": {
            "artifact": AeFromArtifact,
            "resnet": ResNetAE,
            "simple": SimpleConvAE,
            "vqgan": VqGanAe,
        },
        "disc_arch": {"sample": Fcn, "set": SetFcn},
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "gt": GroundTruthLabeller,
            "kmeans": KmeansOnClipEncodings,
            "artifact": LabelFromArtifact,
            "none": NullLabeller,
            "uniform_noise": UniformLabelNoiser,
        },
    }

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        assert isinstance(self.ae_arch, AeFactory)
        assert isinstance(self.disc_arch, PredictorFactory)
        assert isinstance(self.ds, DatasetFactory)
        assert isinstance(self.labeller, Labeller)
        assert isinstance(self.scorer, Scorer)

        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller, self.ae_arch, self.disc_arch))
        dm = self.init_dm(ds, self.labeller)
        ae_pair = self.ae_arch(input_shape=dm.dim_x)
        ae = SplitLatentAe(cfg=self.ae, model=ae_pair, feature_group_slices=dm.feature_group_slices)
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        disc_net, _ = self.disc_arch(
            input_dim=ae.encoding_size.zy, target_dim=1, batch_size=dm.batch_size_tr
        )
        disc = NeuralDiscriminator(model=disc_net, cfg=self.disc)
        score = self.alg.run(dm=dm, ae=ae, disc=disc, evaluator=self.eval, scorer=self.scorer)
        if run is not None:
            # Bar the saving of AeFromArtifact instances to prevent infinite recursion.
            if (self.artifact_name is not None) and (not isinstance(self.ae_arch, AeFromArtifact)):
                ae_config = asdict(self.ae_arch) | {"_target_": full_class_path(self.ae_arch)}
                save_ae_artifact(run=run, model=ae_pair, config=ae_config, name=self.artifact_name)
            run.finish()
        return score
