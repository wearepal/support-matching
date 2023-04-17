from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union, cast
import attr

from conduit.data.datasets.vision import CdtVisionDataset, ImageTform
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch

__all__ = ["NIHChestXRayDataset"]


class NiHSensAttr(Enum):
    gender = "Patient Gender"


class NiHTargetAttr(Enum):
    """
        Fraction of labels that are positive for each thoracic disease:
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
            Pneumothorax          0.04728

        The same as above but conditioned on gender:
            Atelectasis  Cardiomegaly  Consolidation     Edema  Effusion  Emphysema  Fibrosis    Hernia  Infiltration      Mass    Nodule  Pleural_Thickening  Pneumonia  Pneumothorax
    Gender
    F          0.095387      0.030115       0.041021  0.022530  0.120582   0.018573  0.015806  0.002686      0.173575  0.046187  0.054244            0.027532   0.012157      0.052993
    M          0.109031      0.020635       0.042090  0.019009  0.117382   0.025418  0.014446  0.001516      0.180407  0.055715  0.058178            0.032239   0.013230      0.042895
    """

    atelectasis = "Atelectasis"
    cardiomegaly = "Cardiomegaly"
    consolidation = "Consolidation"
    edema = "Edema"
    effusion = "Effusion"
    emphysema = "Emphysema"
    fibrosis = "Fibrosis"
    hernia = "Hernia"
    infiltration = "Infiltration"
    mass = "Mass"
    nodule = "Nodule"
    pleural_thickening = "Pleural_Thickening"
    pneumonia = "Pneumonia"
    pneumothorax = "Pneumothorax"
    no_finding = "No Finding"


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class NIHChestXRayDataset(CdtVisionDataset):
    """ "
    National Institutes of Health Chest X-Ray Dataset
    This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from
    30,805 unique patients. To create these labels, the authors used Natural Language Processing to
    text-mine disease classifications from the associated radiological reports. The labels are
    expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology
    reports are not publicly available but you can find more details on the labeling process in
    `this
    <https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community>`__
    Open Access paper.

    The dataset can be downloaded by following the above link or from `kaggle <https://www.kaggle.com/datasets/nih-chest-xrays/data>`__
    """

    root: Union[Path, str]
    sens_attr: NiHSensAttr = NiHSensAttr.gender
    target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.cardiomegaly
    transform: Optional[ImageTform] = None

    def __attrs_pre_init__(self):
        self.metadata = cast(pd.DataFrame, pd.read_csv(self.root / "Data_Entry_2017.csv"))
        # In the case of Patient Gender, factorize yields the mapping: M -> 0, F -> 1
        s = torch.as_tensor(self.metadata[self.sens_attr.value].factorize()[0], dtype=torch.long)
        findings_str = self.metadata["Finding Labels"].str.split("|")
        self.encoder = MultiLabelBinarizer().fit(findings_str)
        findings_ml = pd.DataFrame(
            self.encoder.transform(findings_str), columns=self.encoder.classes_
        )
        self.metadata = pd.concat((self.metadata, findings_ml), axis=1)
        if self.target_attr is None:
            findings_ml.drop("No Finding", axis=1, inplace=True)
        else:
            findings_ml = findings_ml[self.target_attr.value]
            if self.target_attr is NiHTargetAttr.no_finding:
                findings_ml = 1 - findings_ml
        y = torch.as_tensor(findings_ml.to_numpy(), dtype=torch.long)
        image_index_flat = self.root.glob("*/*/*")
        self.metadata["Image Index"] = sorted(list(image_index_flat))
        x = self.metadata["Image Index"].to_numpy()
        super().__init__(image_dir=self.root, x=x, s=s, y=y, transform=self.transform)
