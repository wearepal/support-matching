from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Tuple, TypedDict, TypeVar

import clip
from conduit.data.datasets.utils import CdtDataLoader, stratified_split
from conduit.data.datasets.vision import CelebA
from conduit.data.datasets.vision.celeba import CelebAttr
from conduit.data.structures import TernarySample
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from shared.configs.arguments import SplitConf
from shared.data.data_module import DataModule
from shared.data.utils import labels_to_group_id

DOWNLOAD_ROOT: Final = "/srv/galene0/shared/models/clip/"
ENCODINGS_FILE: Final = Path("encoded_celeba.npz")
NUM_CLUSTERS: Final = 4


class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


class NpzContent(TypedDict):
    train: npt.NDArray
    train_ids: npt.NDArray
    dep: npt.NDArray
    test: npt.NDArray
    test_ids: npt.NDArray


def main() -> None:
    if ENCODINGS_FILE.exists():
        print("Loading encodings from file...")
        with ENCODINGS_FILE.open("rb") as f:
            enc: NpzContent = np.load(f)
            to_cluster = np.concatenate([enc["train"], enc["dep"]], axis=0)
            to_test = enc["test"]
            test_labels = enc["test_ids"]
            other_data = enc["dep"]

            centroids = np.stack(
                precomputed_centroids(enc["train"], enc["train_ids"], range(NUM_CLUSTERS)), axis=0
            )
        print("Done.")
    else:
        train_enc, train_group_ids, deployment_enc, test_enc, test_group_ids = generate_encodings()
        to_cluster = torch.cat([train_enc, deployment_enc], dim=0).numpy()
        to_test = test_enc.numpy()
        test_labels = test_group_ids.numpy()
        other_data = deployment_enc.numpy()

        centroids = torch.stack(
            precomputed_centroids(train_enc, train_group_ids, range(NUM_CLUSTERS)), dim=0
        ).numpy()

    print("Using kmeans++")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=10)
    kmeans.fit(to_cluster)
    clusters = kmeans.predict(to_test)
    evaluate(test_labels, clusters)

    print("Using pre-computed centroids")
    if centroids.shape[0] < NUM_CLUSTERS:
        print("Using kmeans++ to generate additional initial clusters...")
        additional_centroids, _ = kmeans_plusplus(
            other_data, n_clusters=NUM_CLUSTERS - centroids.shape[0], random_state=0
        )
        centroids = np.concatenate([centroids, additional_centroids], axis=0)
        print("Done.")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init=centroids, n_init=1)
    kmeans.fit(to_cluster)
    clusters = kmeans.predict(to_test)
    evaluate(test_labels, clusters)


T = TypeVar("T", Tensor, npt.NDArray[Any])


def precomputed_centroids(
    train_enc: T, train_group_ids: T, all_group_ids: Iterable[int]
) -> List[T]:
    centroids: List[T] = []
    for group in all_group_ids:
        mask = train_group_ids == group
        if mask.sum() > 0:
            centroid: T = train_enc[mask].mean(0)
            centroids.append(centroid)
    return centroids


def generate_encodings() -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate encodings by putting the data through a pre-trained model."""
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

    to_save: NpzContent = {
        "train": train_enc.numpy(),
        "train_ids": train_group_ids.numpy(),
        "dep": deployment_enc.numpy(),
        "test": test_enc.numpy(),
        "test_ids": test_group_ids.numpy(),
    }
    save_encoding(to_save)
    return train_enc, train_group_ids, deployment_enc, test_enc, test_group_ids


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


def save_encoding(all_encodings: NpzContent) -> None:
    print("Saving encodings to 'encoded_celeba.npz'...")
    np.savez_compressed(ENCODINGS_FILE, **all_encodings)
    print("Done.")


def evaluate(test_group_ids: npt.NDArray, clusters: npt.NDArray) -> None:
    print(f"ARI: {adjusted_rand_score(test_group_ids, clusters)}")
    print(f"AMI: {adjusted_mutual_info_score(test_group_ids, clusters)}")
    print(f"NMI: {normalized_mutual_info_score(test_group_ids, clusters)}")
    print(f"Accuracy: {compute_accuracy(test_group_ids, clusters)}")


def compute_accuracy(
    test_group_ids: npt.NDArray[np.int64], clusters: npt.NDArray[np.int64]
) -> float:
    # in order to solve the assignment problem, we find the assignment that maximizes counts
    counts = count_cooccurrances(test_group_ids, clusters)
    row_ind, col_ind = linear_sum_assignment(counts, maximize=True)
    num_corectly_assigned = counts[row_ind, col_ind].sum()
    return num_corectly_assigned / test_group_ids.shape[0]


def count_cooccurrances(
    test_group_ids: npt.NDArray[np.int64], clusters: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Count how often every possible pair of group ID and cluster ID co-occur."""
    counts: Dict[Tuple[int, int], int] = {}
    max_group = 0
    max_cluster = 0
    for group in np.unique(test_group_ids):
        for cluster in np.unique(clusters):
            counts[(group, cluster)] = np.count_nonzero(
                (test_group_ids == group) & (clusters == cluster)
            )
            if cluster > max_cluster:
                max_cluster = cluster
        if group > max_group:
            max_group = group
    counts_np = np.zeros((max_group + 1, max_cluster + 1), dtype=np.int64)
    for (group, cluster), count in counts.items():
        counts_np[group, cluster] = count
    return counts_np


if __name__ == "__main__":
    main()
