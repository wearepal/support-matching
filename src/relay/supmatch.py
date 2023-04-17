from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, ClassVar

from src.hydra_confs.camelyon17.conf import Camelyon17Conf
from src.hydra_confs.celeba.conf import CelebAConf
from src.hydra_confs.cmnist.conf import ColoredMNISTConf
from src.hydra_confs.nih.conf import NIHChestXRayDatasetConf
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Option

from src.arch.predictors.fcn import Fcn, SetFcn
from src.arch.autoencoder import AeFromArtifact, ResNetAE, SimpleConvAE, VqGanAe
from src.algs import SupportMatching
from src.algs.adv import Evaluator, NeuralScorer, NullScorer, Scorer
from src.arch.autoencoder import AeFactory, AeFromArtifact, AePair, save_ae_artifact
from src.models import SplitLatentAe
from src.models.discriminator import NeuralDiscriminator
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    NullLabeller,
    UniformLabelNoiser,
)

from .base import BaseRelay

__all__ = ["SupMatchRelay"]


@dataclass(eq=False, kw_only=True)
class SupMatchRelay(BaseRelay):
    # defaults: List[Any] = field(default_factory=lambda: [{"split": "random"}])
    alg: SupportMatching
    ae: SplitLatentAe
    ae_arch: Any
    ds: Any
    disc_arch: Any
    disc: NeuralDiscriminator
    eval: Evaluator
    labeller: Any
    scorer: Any  # Union[NeuralScorer, NullScorer]
    artifact_name: Optional[str] = None

    options: ClassVar[Dict[str, type]] = BaseRelay.options | {
        "scorer": {"neural": NeuralScorer, "none": NullScorer},
        "ds": {
            "cmnist": ColoredMNISTConf,
            "celeba": CelebAConf,
            "camelyon17": Camelyon17Conf,
            "nih": NIHChestXRayDatasetConf,
        },
        "ae_arch": {
            "artifact": AeFromArtifact,
            "resnet": ResNetAE,
            "simple": SimpleConvAE,
            "vqgan": VqGanAe,
        },
        "disc_arch": {
            "sample": Fcn,
            "set": SetFcn,
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

    @classmethod
    @implements(BaseRelay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        ds: List[Option],
        ae_arch: List[Option],
        disc: List[Option],
        disc_arch: List[Option],
        labeller: List[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            ae=[Option(SplitLatentAe, name="base")],
            ae_arch=ae_arch,
            alg=[Option(SupportMatching, name="base")],
            disc=disc,
            disc_arch=disc_arch,
            ds=ds,
            eval=[Option(Evaluator, name="base")],
            labeller=labeller,
            scorer=[Option(NeuralScorer, name="neural"), Option(NullScorer, name="none")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        run = self.init_wandb(raw_config, self.labeller, self.ae_arch, self.disc_arch)
        dm = self.init_dm()
        alg: SupportMatching = instantiate(self.alg)
        ae_factory: AeFactory = instantiate(self.ae_arch)
        ae_pair: AePair = ae_factory(input_shape=dm.dim_x)
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
        disc: NeuralDiscriminator = instantiate(self.disc, _partial_=True)(model=disc_net)
        evaluator: Evaluator = instantiate(self.eval)
        scorer: Scorer = instantiate(self.scorer)
        score = alg.run(dm=dm, ae=ae, disc=disc, evaluator=evaluator, scorer=scorer)
        if run is not None:
            # Bar the saving of AeFromArtifact instances to prevent infinite recursion.
            if (self.artifact_name is not None) and (not isinstance(ae_factory, AeFromArtifact)):
                save_ae_artifact(
                    run=run, model=ae_pair, config=self.ae_arch, name=self.artifact_name
                )
            run.finish()
        return score
