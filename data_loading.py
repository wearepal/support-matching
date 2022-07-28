from enum import Enum
from typing import Final

from conduit.data.datasets.utils import stratified_split
from conduit.data.datasets.vision import CelebA
from conduit.data.datasets.vision.celeba import CelebAttr

from shared.configs.arguments import SplitConf
from shared.data.data_module import DataModule

__all__ = ["get_data", "DOWNLOAD_ROOT", "CLIPVersion", "MODEL_PATH", "CLIP_VER"]

DOWNLOAD_ROOT: Final = "/srv/galene0/shared/models/clip/"
MODEL_PATH: Final = "./finetuned.pt"


class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


CLIP_VER: Final = CLIPVersion.RN50
# CLIP_VER: Final = CLIPVersion.ViT_L14


def get_data(transforms, batch_size_tr: int) -> DataModule[CelebA]:
    data_settings = SplitConf(
        # data_prop=0.01,  # for testing
        train_transforms=transforms,
        dep_transforms=transforms,
        test_transforms=transforms,
        # subsampling
        train_subsampling_props={0: {1: 0.3}, 1: {0: 0.0}},
        dep_subsampling_props={0: {0: 0.7, 1: 0.4}, 1: {0: 0.2}},
    )
    return get_dm(
        data_settings,
        superclass=CelebAttr.Smiling,
        subclass=CelebAttr.Male,
        batch_size_tr=batch_size_tr,
    )


def get_dm(
    split_config: SplitConf, superclass: CelebAttr, subclass: CelebAttr, batch_size_tr: int
) -> DataModule[CelebA]:
    root = DataModule.find_data_dir()
    all_data = CelebA(root=root, download=False, superclass=superclass, subclass=subclass)
    if split_config.data_prop is not None:
        print("Making data smaller...", flush=True)
        all_data = stratified_split(all_data, default_train_prop=split_config.data_prop).train
        print("Done.")
    splits = DataModule._generate_splits(dataset=all_data, split_config=split_config)
    print("Done.")
    return DataModule(
        train=splits.train,
        deployment=splits.deployment,
        test=splits.test,
        batch_size_tr=batch_size_tr,
    )
