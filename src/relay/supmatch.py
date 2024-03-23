from dataclasses import asdict, dataclass, field
import math
from typing import Any, ClassVar

from loguru import logger
from ranzen import some

from src.algs import SupportMatching
from src.algs.adv import Evaluator, NeuralScorer, NullScorer, Scorer
from src.algs.base import NaNLossError
from src.arch.autoencoder import (
    AeFactory,
    AeFromArtifact,
    ResNetAE,
    SimpleAE,
    SimpleConvAE,
    VqGanAe,
    save_ae_artifact,
)
from src.arch.predictors.base import PredictorFactory
from src.arch.predictors.fcn import Fcn, SetFcn
from src.data.common import DatasetFactory
from src.data.factories import ACSCfg, NICOPPCfg
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
from src.models import OptimizerCfg, SplitLatentAe
from src.models.autoencoder import SplitAeCfg
from src.models.discriminator import DiscOptimizerCfg, NeuralDiscriminator
from src.utils import full_class_path

from .base import BaseRelay

__all__ = ["SupMatchRelay"]


@dataclass(eq=False, kw_only=True)
class SupMatchRelay(BaseRelay):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"ae_arch": "simple"},
            {"ds": "cmnist"},
            {"disc_arch": "set"},
            {"labeller": "none"},
            {"scorer": "none"},
            {"split": "random"},
        ]
    )
    alg: SupportMatching = field(default_factory=SupportMatching)
    ae: SplitAeCfg = field(default_factory=SplitAeCfg)
    ae_opt: OptimizerCfg = field(default_factory=OptimizerCfg)
    ae_arch: AeFactory
    ds: DatasetFactory
    disc_arch: PredictorFactory
    disc: DiscOptimizerCfg = field(default_factory=DiscOptimizerCfg)
    eval: Evaluator = field(default_factory=Evaluator)
    labeller: Labeller
    scorer: Scorer
    artifact_name: str | None = None
    """Save model weights under this name."""

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "scorer": {"neural": NeuralScorer, "none": NullScorer},
        "ds": {
            "acs": ACSCfg,
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
            "fcn": SimpleAE,
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

    def run(self, raw_config: dict[str, Any] | None = None) -> float | None:
        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller, self.ae_arch, self.disc_arch))
        dm = self.init_dm(ds, self.labeller, device=self.alg.device)
        # dm.print_statistics()
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
        disc_net, _ = self.disc_arch(
            input_dim=ae.encoding_size.zy, target_dim=1, batch_size=dm.batch_size_tr
        )
        disc = NeuralDiscriminator(model=disc_net, opt=self.disc, criterion=self.disc.criterion)
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
