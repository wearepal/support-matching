from dataclasses import asdict
import math
from typing import Any, ClassVar, Optional

from attrs import define, field
from loguru import logger
from ranzen import some

from src.algs import SupportMatching
from src.algs.adv import Evaluator, NeuralScorer, NullScorer, Scorer
from src.algs.base import NaNLossError
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
from src.models.autoencoder import SplitAeOptimizerCfg
from src.models.discriminator import DiscOptimizerCfg, NeuralDiscriminator
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
    ae: SplitAeOptimizerCfg = field(default=SplitAeOptimizerCfg)
    ae_arch: Any  # AeFactory
    ds: Any  # DatasetFactory
    disc_arch: Any  # PredictorFactory
    disc: DiscOptimizerCfg = field(default=DiscOptimizerCfg)
    eval: Evaluator = field(default=Evaluator)
    labeller: Any  # Labeller
    scorer: Any  # Scorer
    artifact_name: Optional[str] = None
    """Save model weights under this name."""

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
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

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> Optional[float]:
        assert isinstance(self.ae_arch, AeFactory)
        assert isinstance(self.disc_arch, PredictorFactory)
        assert isinstance(self.ds, DatasetFactory)
        assert isinstance(self.labeller, Labeller)
        assert isinstance(self.scorer, Scorer)

        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller, self.ae_arch, self.disc_arch))
        dm = self.init_dm(ds, self.labeller, device=self.alg.device)
        # dm.print_statistics()
        input_shape: tuple[int, int, int] = dm.dim_x  # type: ignore
        ae_pair = self.ae_arch(input_shape=input_shape)
        ae = SplitLatentAe(opt=self.ae, model=ae_pair, feature_group_slices=dm.feature_group_slices)
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        disc_net, _ = self.disc_arch(
            input_dim=ae.encoding_size.zy, target_dim=1, batch_size=dm.batch_size_tr
        )
        disc = NeuralDiscriminator(model=disc_net, opt=self.disc)
        try:
            score = self.alg.fit_evaluate_score(
                dm=dm, ae=ae, disc=disc, evaluator=self.eval, scorer=self.scorer
            )
        except NaNLossError:
            logger.info("Stopping due to NaN loss")
            return -math.inf
        if some(run):
            if self.artifact_name is not None:
                if isinstance(self.ae_arch, AeFromArtifact):
                    # An `AeFromArtifact` doesn't know how to instantiate the model architecture,
                    # so we have to take the information from the loaded config.
                    ae_config = self.ae_arch.factory_config
                else:
                    ae_config = asdict(self.ae_arch) | {"_target_": full_class_path(self.ae_arch)}
                save_ae_artifact(
                    run=run, model=ae_pair, factory_config=ae_config, name=self.artifact_name
                )
            run.finish()
        return score
