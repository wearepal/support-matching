from __future__ import annotations
from typing import Any, ClassVar, Dict, Optional

from attrs import define, field
import torch.nn as nn

from src.algs.fs import Dro, Erm, FsAlg, Gdro, Jtt, LfF, SdErm
from src.arch import BackboneFactory, PredictorFactory
from src.arch.backbones.vision import DenseNet, ResNet, SimpleCNN
from src.arch.predictors.fcn import Fcn
from src.hydra_confs.camelyon17.conf import Camelyon17Conf
from src.hydra_confs.celeba.conf import CelebAConf
from src.hydra_confs.cmnist.conf import ColoredMNISTConf
from src.hydra_confs.nih.conf import NIHChestXRayDatasetConf
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    NullLabeller,
    UniformLabelNoiser,
)

from .base import BaseRelay

__all__ = ["FsRelay"]


@define(eq=False, kw_only=True)
class FsRelay(BaseRelay):
    alg: Any
    ds: Any
    backbone: Any
    predictor: Fcn = field(default=Fcn)
    labeller: Any

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "cmnist": ColoredMNISTConf,
            "celeba": CelebAConf,
            "camelyon17": Camelyon17Conf,
            "nih": NIHChestXRayDatasetConf,
        },
        "backbone": {
            "densenet": DenseNet,
            "resnet": ResNet,
            "simple": SimpleCNN,
        },
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "gt": GroundTruthLabeller,
            "kmeans": KmeansOnClipEncodings,
            "artifact": LabelFromArtifact,
            "none": NullLabeller,
            "uniform_noise": UniformLabelNoiser,
        },
        "alg": {
            "dro": Dro,
            "erm": Erm,
            "gdro": Gdro,
            "jtt": Jtt,
            "lff": LfF,
            "sd": SdErm,
        },
    }

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Any:
        run = self.wandb.init(raw_config, (self.labeller, self.backbone, self.predictor))
        dm = self.init_dm(self.ds, self.labeller)
        alg: FsAlg = self.alg
        backbone_fn: BackboneFactory = self.backbone
        predictor_fn: PredictorFactory = self.predictor
        backbone, out_dim = backbone_fn(input_dim=dm.dim_x[0])
        predictor, _ = predictor_fn(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)
        result = alg.run(dm=dm, model=model)
        if run is not None:
            run.finish()  # type: ignore
        return result
