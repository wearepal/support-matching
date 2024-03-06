from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.data.common import DatasetFactory
from src.data.factories import ACSCfg
from src.data.nih import NIHChestXRayDatasetCfg
from src.data.utils import resolve_device
from src.hydra_confs.datasets import Camelyon17Cfg, CelebACfg, ColoredMNISTCfg
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    ClipClassifier,
    KmeansOnClipEncodings,
    Labeller,
    UniformLabelNoiser,
)

from .base import BaseRelay

__all__ = ["LabelRelay"]


@dataclass(eq=False, kw_only=True)
class LabelRelay(BaseRelay):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"ds": "cmnist"},
            {"labeller": "uniform_noise"},
            {"split": "random"},
        ]
    )

    ds: Any  # CdtDataset
    labeller: Any  # Labeller
    gpu: int = 0

    options: ClassVar[dict[str, dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "acs": ACSCfg,
            "cmnist": ColoredMNISTCfg,
            "celeba": CelebACfg,
            "camelyon17": Camelyon17Cfg,
            "nih": NIHChestXRayDatasetCfg,
        },
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "classifier": ClipClassifier,
            "kmeans": KmeansOnClipEncodings,
            "uniform_noise": UniformLabelNoiser,
        },
    }

    def run(self, raw_config: dict[str, Any] | None = None) -> float | None:
        assert isinstance(self.ds, DatasetFactory)
        assert isinstance(self.labeller, Labeller)

        ds = self.ds()
        run = self.wandb.init(raw_config, (ds, self.labeller))
        device = resolve_device(self.gpu)
        self.init_dm(ds, self.labeller, device=device)
        if run is not None:
            run.finish()
