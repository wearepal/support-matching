from typing import Any, ClassVar, Dict, Optional

from attrs import define, field

from src.hydra_confs.camelyon17.conf import Camelyon17Conf
from src.hydra_confs.celeba.conf import CelebAConf
from src.hydra_confs.cmnist.conf import ColoredMNISTConf
from src.hydra_confs.nih.conf import NIHChestXRayDatasetConf
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    ClipClassifier,
    KmeansOnClipEncodings,
    Labeller,
    UniformLabelNoiser,
)

from .base import BaseRelay

__all__ = ["LabelRelay"]


@define(eq=False, kw_only=True)
class LabelRelay(BaseRelay):
    defaults: list[Any] = field(
        default=[{"ds": "cmnist"}, {"labeller": "uniform_noise"}, {"split": "random"}]
    )

    ds: Any  # CdtDataset
    labeller: Any  # Labeller

    options: ClassVar[Dict[str, Dict[str, type]]] = BaseRelay.options | {
        "ds": {
            "cmnist": ColoredMNISTConf,
            "celeba": CelebAConf,
            "camelyon17": Camelyon17Conf,
            "nih": NIHChestXRayDatasetConf,
        },
        "labeller": {
            "centroidal_noise": CentroidalLabelNoiser,
            "classifier": ClipClassifier,
            "kmeans": KmeansOnClipEncodings,
            "uniform_noise": UniformLabelNoiser,
        },
    }

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        assert isinstance(self.labeller, Labeller)

        run = self.wandb.init(raw_config, (self.labeller,))
        self.init_dm(self.ds, self.labeller)
        if run is not None:
            run.finish()
