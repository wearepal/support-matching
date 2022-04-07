from enum import Enum
from typing import Final, List, Tuple

from conduit.data.datasets.utils import CdtDataLoader, stratified_split
from conduit.data.datasets.vision import CelebA
from conduit.data.structures import TernarySample
import numpy.typing as npt
import torch
from torch import Tensor, nn

import clip
from shared.configs.arguments import SplitConf
from shared.data.data_module import DataModule
from shared.data.utils import labels_to_group_id


DOWNLOAD_ROOT: Final = "/srv/galene0/shared/models/clip/"

class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


def main() -> None:
    dm = get_data(SplitConf())

    model, transforms = clip.load(
        name=CLIPVersion.ViT_B32.value, device="cpu", download_root=DOWNLOAD_ROOT
    )
    visual_model = model.visual
    out_dim = visual_model.output_dim
    encoded = encode(visual_model, dm.train_dataloader())
    breakpoint()


def encode(model: nn.Module, dl: CdtDataLoader[TernarySample[Tensor]]) -> Tuple[Tensor, Tensor]:
    encoded: List[Tensor] = []
    group_ids: List[Tensor] = []
    for sample in dl:
        encoded.append(model(sample.x))
        group_ids.append(labels_to_group_id(s=sample.s, y=sample.y, s_count=2))
    return torch.cat(encoded, dim=0), torch.cat(group_ids, dim=0)


def get_centroids(data: Tensor) -> Tensor:
    pass


def get_data(split_config: SplitConf) -> DataModule[CelebA]:
    root = DataModule.find_data_dir()
    all_data = CelebA(root=root, download=False)
    if split_config.data_prop is not None:
        all_data = stratified_split(all_data, default_train_prop=split_config.data_prop).train
    splits = DataModule._generate_splits(dataset=all_data, split_config=split_config)
    return DataModule(
        train=splits.train,
        deployment=splits.deployment,
        test=splits.test,
        batch_size_tr=256,
    )


if __name__ == "__main__":
    main()
