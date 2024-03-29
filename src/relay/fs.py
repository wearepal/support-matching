from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch.nn as nn

from src.algs.fs import Dro, Erm, FsAlg, Gdro, Jtt, LfF, SdErm
from src.arch import BackboneFactory
from src.arch.backbones import DenseNet, LinearResNet, ResNet, SimpleCNN
from src.arch.predictors.fcn import Fcn
from src.data import DatasetFactory, NICOPPCfg, NIHChestXRayDatasetCfg
from src.data.factories import ACSCfg
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

from .base import BaseRelay

__all__ = ["FsRelay"]


@dataclass(eq=False, kw_only=True)
class FsRelay(BaseRelay):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"alg": "erm"},
            {"ds": "cmnist"},
            {"backbone": "simple"},
            {"labeller": "none"},
            {"split": "random"},
        ]
    )

    alg: FsAlg
    ds: DatasetFactory
    backbone: BackboneFactory
    predictor: Fcn = field(default_factory=Fcn)
    labeller: Labeller

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "acs": ACSCfg,
            "cmnist": ColoredMNISTCfg,
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
            "nicopp": NICOPPCfg,
        },
        "backbone": {
            "densenet": DenseNet,
            "resnet": ResNet,
            "simple": SimpleCNN,
            "linear": LinearResNet,
        },
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

    def run(self, raw_config: dict[str, Any] | None = None) -> float | None:
        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller, self.backbone, self.predictor))
        dm = self.init_dm(ds, self.labeller, device=self.alg.device)
        backbone, out_dim = self.backbone(input_dim=dm.dim_x[0])
        predictor, _ = self.predictor(input_dim=out_dim, target_dim=dm.card_y)
        model = nn.Sequential(backbone, predictor)
        result = self.alg.run(dm=dm, model=model)
        if run is not None:
            run.finish()
        return result
