from enum import Enum
from pathlib import Path
from typing import Optional, Union, cast

from conduit.data.datasets.utils import ImageTform
from conduit.data.datasets.vision.base import CdtVisionDataset
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch

__all__ = ["NIHChestXRayDataset"]


class NiHSensAttr(Enum):
    GENDER = "Patient Gender"


class NiHTargetAttr(Enum):
    ATELECTASIS = "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    EDEMA = "Edema"
    EFFUSION = "Effusion"
    EMPHYSEMA = "Emphysema"
    FIBROSIS = "Fibrosis"
    HERNIA = "Hernia"
    INFILTRATION = "Infiltration"
    MASS = "Mass"
    NODULE = "Nodule"
    PLEURAL_THICKENING = "Pleural_Thickening"
    PNEUMONIA = "Pneumonia"
    PNEUMOTHORAX = "Pneumothorax"


class NIHChestXRayDataset(CdtVisionDataset):
    def __init__(
        self,
        root: Union[Path, str],
        sens_attr: NiHSensAttr = NiHSensAttr.GENDER,
        target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.CARDIOMEGALY,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = cast(pd.DataFrame, pd.read_csv(self.root / "Data_Entry_2017.csv"))
        s = torch.as_tensor(self.metadata[sens_attr.value].factorize()[0], dtype=torch.long)
        y_str = self.metadata["Finding Labels"].str.split("|")
        self.encoder = MultiLabelBinarizer().fit(y_str)
        y_encoded = pd.DataFrame(self.encoder.transform(y_str), columns=self.encoder.classes_)
        y_encoded.drop("No Finding", axis=1, inplace=True)
        if target_attr is not None:
            y_encoded = y_encoded[target_attr.value]
        y = torch.as_tensor(y_encoded.to_numpy(), dtype=torch.long)
        image_index_flat = self.root.glob("*/*/*")
        self.metadata["Image Index"] = sorted(list(image_index_flat))
        x = self.metadata["Image Index"].to_numpy()
        super().__init__(image_dir=self.root, x=x, s=s, y=y, transform=transform)
