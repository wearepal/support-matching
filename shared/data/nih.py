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
    """
    Fraction of labels that are positive for each 'finding':
        Atelectasis           0.103095
        Cardiomegaly          0.024759
        Consolidation         0.041625
        Edema                 0.020540
        Effusion              0.118775
        Emphysema             0.022440
        Fibrosis              0.015037
        Hernia                0.002025
        Infiltration          0.177435
        Mass                  0.051570
        Nodule                0.056466
        Pleural_Thickening    0.030191
        Pneumonia             0.012763
        Pneumothorax          0.047289

    The same as above but conditioned on 'Patient Gender':
        Atelectasis  Cardiomegaly  Consolidation     Edema  Effusion  Emphysema  Fibrosis    Hernia  Infiltration      Mass    Nodule  Pleural_Thickening  Pneumonia  Pneumothorax
Gender
F          0.095387      0.030115       0.041021  0.022530  0.120582   0.018573  0.015806  0.002686      0.173575  0.046187  0.054244            0.027532   0.012157      0.052993
M          0.109031      0.020635       0.042090  0.019009  0.117382   0.025418  0.014446  0.001516      0.180407  0.055715  0.058178            0.032239   0.013230      0.042895
    """
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
        findings_str = self.metadata["Finding Labels"].str.split("|")
        self.encoder = MultiLabelBinarizer().fit(findings_str)
        findings_ml = pd.DataFrame(self.encoder.transform(findings_str), columns=self.encoder.classes_)
        findings_ml.drop("No Finding", axis=1, inplace=True)
        self.metadata = pd.concat((self.metadata, findings_ml), axis=1)
        if target_attr is not None:
            findings_ml = findings_ml[target_attr.value]
        y = torch.as_tensor(findings_ml.to_numpy(), dtype=torch.long)
        image_index_flat = self.root.glob("*/*/*")
        self.metadata["Image Index"] = sorted(list(image_index_flat))
        x = self.metadata["Image Index"].to_numpy()
        super().__init__(image_dir=self.root, x=x, s=s, y=y, transform=transform)
