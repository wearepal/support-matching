from enum import Enum
from typing import Dict, Final, Iterable, List, Tuple

import clip
from conduit.data.datasets.utils import CdtDataLoader, stratified_split
from conduit.data.datasets.vision import CelebA
from conduit.data.datasets.vision.celeba import CelebAttr
from conduit.data.structures import TernarySample
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from shared.configs.arguments import SplitConf
from shared.data.data_module import DataModule
from shared.data.utils import labels_to_group_id
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

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
    print("Loading CLIP model (downloading if needed)...", flush=True)
    model, transforms = clip.load(
        name=CLIPVersion.ViT_L14.value, device="cpu", download_root=DOWNLOAD_ROOT
    )
    print("Done.")
    visual_model = model.visual
    visual_model.to("cuda:0")
    # out_dim = visual_model.output_dim

    data_settings = SplitConf(
        # data_prop=0.01,  # for testing
        train_transforms=transforms,
        dep_transforms=transforms,
        test_transforms=transforms,
    )
    dm = get_data(data_settings, superclass=CelebAttr.Smiling, subclass=CelebAttr.Male)

    train_enc, train_group_ids = encode(visual_model, dm.train_dataloader(eval=True))
    deployment_enc, _ = encode(visual_model, dm.deployment_dataloader(eval=True))
    test_enc, test_group_ids = encode(visual_model, dm.test_dataloader())
    del model
    del visual_model
    torch.cuda.empty_cache()

    to_save = {
        "train": train_enc,
        "train_ids": train_group_ids,
        "dep": deployment_enc,
        "test": test_enc,
        "test_ids": test_group_ids,
    }
    save_encoding(to_save)

    centroids = get_centroids(train_enc, train_group_ids, deployment_enc, [0, 1, 2, 3])

    print("Using kmeans++")
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10)
    kmeans.fit(torch.cat([train_enc, deployment_enc], dim=0).numpy())
    clusters = kmeans.predict(test_enc.numpy())
    evaluate(test_group_ids, clusters)

    print("Using pre-computed centroids")
    kmeans = KMeans(n_clusters=4, init=centroids.numpy(), n_init=1)
    kmeans.fit(torch.cat([train_enc, deployment_enc], dim=0).numpy())
    clusters = kmeans.predict(test_enc.numpy())
    evaluate(test_group_ids, clusters)


def get_data(
    split_config: SplitConf, superclass: CelebAttr, subclass: CelebAttr
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
        batch_size_tr=150,
    )


def encode(model: nn.Module, dl: CdtDataLoader[TernarySample[Tensor]]) -> Tuple[Tensor, Tensor]:
    encoded: List[Tensor] = []
    group_ids: List[Tensor] = []
    print("start encoding...", flush=True)
    with torch.set_grad_enabled(False):
        for sample in tqdm(dl, total=len(dl)):
            enc = model(sample.x.to("cuda:0", non_blocking=True)).detach()
            # normalize so we're doing cosine similarity
            encoded.append(F.normalize(enc, dim=1, p=2).cpu())
            group_ids.append(labels_to_group_id(s=sample.s, y=sample.y, s_count=2))
    print("done.")
    return torch.cat(encoded, dim=0), torch.cat(group_ids, dim=0)


def save_encoding(all_encodings: Dict[str, Tensor]) -> None:
    print("Saving encodings to 'encoded_celeba.npz'...")
    np.savez_compressed("encoded_celeba.npz", **{k: v.numpy() for k, v in all_encodings.items()})
    print("Done.")


def get_centroids(
    train_enc: Tensor, train_group_ids: Tensor, deployment_enc: Tensor, all_group_ids: Iterable[int]
) -> Tensor:
    centroids: List[Tensor] = []
    for group in all_group_ids:
        mask = train_group_ids == group
        if mask.sum() > 0:
            centroids.append(train_enc[mask].mean(dim=0))
        else:
            del deployment_enc
            raise NotImplementedError(
                "For some of the groups we obviously have no samples from the training set."
                "In this case, have to make use of the deployment set somehow."
            )
    return torch.stack(centroids, dim=0)


def evaluate(test_group_ids: Tensor, clusters: npt.NDArray) -> None:
    print(f"ARI: {adjusted_rand_score(test_group_ids.numpy(), clusters)}")
    print(f"AMI: {adjusted_mutual_info_score(test_group_ids.numpy(), clusters)}")
    print(f"NMI: {normalized_mutual_info_score(test_group_ids.numpy(), clusters)}")


if __name__ == "__main__":
    main()
