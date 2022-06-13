from enum import Enum, auto
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
    CARDIOMEGALY = auto()


class NIHChestXRayDataset(CdtVisionDataset):
    def __init__(
        self,
        root: Union[Path, str],
        sens_attr: NiHSensAttr = NiHSensAttr.GENDER,
        target_attr: Optional[NiHTargetAttr] = NiHTargetAttr.CARDIOMEGALY,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = cast(pd.DataFrame, pd.read_csv(self.root / "Data_Entry_2017"))
        s = torch.as_tensor(
            self.metadata[sens_attr.value].factorize()[0].to_numpy(), dtype=torch.long
        )
        y_pd = self.metadata["Finding Labels"]
        y_pd = y_pd.str.split("|", expand=True)
        mlb = MultiLabelBinarizer().fit(y_pd)
        y_pd = pd.DataFrame(mlb.transform(y_pd), columns=mlb.classes_)
        if target_attr is not None:
            y_pd = y_pd[target_attr.value]
        y = torch.as_tensor(y_pd.to_numpy(), dtype=torch.long)
        image_index_flat = self.root.glob("**/*")
        self.metadata["Image Index"] = sorted(list(image_index_flat))
        x = self.metadata["Image Index"].to_numpy()
        super().__init__(image_dir=self.root, x=x, s=s, y=y, transform=transform)
