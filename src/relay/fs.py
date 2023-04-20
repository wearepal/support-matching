from typing import Any, ClassVar, Optional, cast

from attrs import define, field
import torch.nn as nn

from src.algs.fs import Dro, Erm, FsAlg, Gdro, Jtt, LfF, SdErm
from src.arch import BackboneFactory
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
    Labeller,
    NullLabeller,
    UniformLabelNoiser,
)

from .base import BaseRelay

__all__ = ["FsRelay"]


@define(eq=False, kw_only=True)
class FsRelay(BaseRelay):
    defaults: list[Any] = field(
        default=[
            {"alg": "erm"},
            {"ds": "cmnist"},
            {"backbone": "simple"},
            {"labeller": "none"},
            {"split": "random"},
        ]
    )

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
        "backbone": {"densenet": DenseNet, "resnet": ResNet, "simple": SimpleCNN},
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "gt": GroundTruthLabeller,
            "kmeans": KmeansOnClipEncodings,
            "artifact": LabelFromArtifact,
            "none": NullLabeller,
            "uniform_noise": UniformLabelNoiser,
        },
        "alg": {"dro": Dro, "erm": Erm, "gdro": Gdro, "jtt": Jtt, "lff": LfF, "sd": SdErm},
    }

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> Optional[float]:
        assert isinstance(self.alg, FsAlg)
        assert isinstance(self.backbone, BackboneFactory)
        self.labeller = cast(Labeller, self.labeller)  # just a Protocol

        run = self.wandb.init(raw_config, (self.labeller, self.backbone, self.predictor))
        dm = self.init_dm(self.ds, self.labeller)
        backbone, out_dim = self.backbone(input_dim=dm.dim_x[0])
        predictor, _ = self.predictor(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)
        result = self.alg.run(dm=dm, model=model)
        if run is not None:
            run.finish()
        return result
