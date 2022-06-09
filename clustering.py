from enum import Enum
from pathlib import Path
from typing import Dict, Final, Iterable, List, NamedTuple, Tuple, TypedDict

import faiss
import clip
from conduit.data.datasets.utils import CdtDataLoader, stratified_split
from conduit.data.datasets.vision import CelebA
from conduit.data.datasets.vision.celeba import CelebAttr
from conduit.data.structures import TernarySample
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
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
USE_FFT: Final = True


class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


class Encodings(NamedTuple):
    """Result of encoding the data."""

    to_cluster: npt.NDArray  # This is passed directly to sklearn's kmeans algorithm.
    train: Tensor  # This is used to compute the pre-defined starting points.
    train_labels: Tensor  # Same as above.
    dep: Tensor  # This is used to compute the other starting points.
    test: npt.NDArray  # This is used for evaluation.
    test_labels: npt.NDArray[np.int32]  # Same as above.


def main() -> None:
    if ENCODINGS_FILE.exists():
        enc = load_encodings(ENCODINGS_FILE)
    else:
        enc = generate_encodings()

    if False:
        print("Using kmeans++")
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=10)
        kmeans.fit(enc.to_cluster)
        clusters = kmeans.predict(enc.test)
        evaluate(enc.test_labels, clusters)

    print("Using pre-computed centroids")
    # known_group_ids = range(NUM_CLUSTERS)
    known_group_ids = [0, 1, 3]
    centroids: Tensor = precomputed_centroids(enc.train, enc.train_labels, known_group_ids)
    if centroids.shape[0] < NUM_CLUSTERS:
        print(f"Need additional clusters: {NUM_CLUSTERS - centroids.shape[0]}")
        if USE_FFT:
            print("Using furthest-first traversal to generate additional initial clusters...")
            centroids_np = furthest_first_traversal(enc.dep, centroids, NUM_CLUSTERS).numpy()
        else:
            print("Using kmeans++ to generate additional initial clusters...")
            additional_centroids, _ = kmeans_plusplus(
                enc.dep.numpy(), n_clusters=NUM_CLUSTERS - centroids.shape[0], random_state=0
            )
            centroids_np = np.concatenate([centroids.numpy(), additional_centroids], axis=0)
        print("Done.")
    else:
        centroids_np = centroids.numpy()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init=centroids_np, n_init=1)
    kmeans.fit(enc.to_cluster)
    clusters = kmeans.predict(enc.test)
    evaluate(enc.test_labels, clusters)


def precomputed_centroids(
    train_enc: Tensor, train_group_ids: Tensor, all_group_ids: Iterable[int]
) -> Tensor:
    """Determine centroids from group IDs."""
    centroids: List[Tensor] = []
    for group in all_group_ids:
        mask = train_group_ids == group
        if mask.count_nonzero() > 0:
            centroids.append(train_enc[mask].mean(0, keepdim=True))
    return torch.cat(centroids, dim=0)


def generate_encodings() -> Encodings:
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
    return Encodings(
        to_cluster=torch.cat([train_enc, deployment_enc], dim=0).numpy(),
        train=train_enc,
        train_labels=train_group_ids,
        dep=deployment_enc,
        test=test_enc.numpy(),
        test_labels=test_group_ids.numpy(),
    )


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


class NpzContent(TypedDict):
    """Content of the npz file (which is basically a dictionary)."""

    train: npt.NDArray
    train_ids: npt.NDArray
    dep: npt.NDArray
    test: npt.NDArray
    test_ids: npt.NDArray[np.int32]


def save_encoding(all_encodings: NpzContent) -> None:
    print("Saving encodings to 'encoded_celeba.npz'...")
    np.savez_compressed(ENCODINGS_FILE, **all_encodings)
    print("Done.")


def load_encodings(fpath: Path) -> Encodings:
    print("Loading encodings from file...")
    with fpath.open("rb") as f:
        loaded: NpzContent = np.load(f)
        enc = Encodings(
            train=torch.from_numpy(loaded["train"]),
            train_labels=torch.from_numpy(loaded["train_ids"]),
            dep=torch.from_numpy(loaded["dep"]),
            test=loaded["test"],
            test_labels=loaded["test_ids"],
            to_cluster=np.concatenate([loaded["train"], loaded["dep"]], axis=0),
        )
    print("Done.")
    return enc


def evaluate(test_group_ids: npt.NDArray[np.int32], clusters: npt.NDArray[np.int32]) -> None:
    print(f"ARI: {adjusted_rand_score(test_group_ids, clusters)}")
    print(f"AMI: {adjusted_mutual_info_score(test_group_ids, clusters)}")
    print(f"NMI: {normalized_mutual_info_score(test_group_ids, clusters)}")
    print(f"Accuracy: {compute_accuracy(test_group_ids, clusters)}")


def compute_accuracy(
    test_group_ids: npt.NDArray[np.int32], clusters: npt.NDArray[np.int32]
) -> float:
    # in order to solve the assignment problem, we find the assignment that maximizes counts
    counts = count_cooccurrances(test_group_ids, clusters)
    row_ind, col_ind = linear_sum_assignment(counts, maximize=True)
    num_corectly_assigned = counts[row_ind, col_ind].sum()
    return num_corectly_assigned / test_group_ids.shape[0]


def count_cooccurrances(
    test_group_ids: npt.NDArray[np.int32], clusters: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
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
    counts_np = np.zeros((max_group + 1, max_cluster + 1), dtype=np.int32)
    for (group, cluster), count in counts.items():
        counts_np[group, cluster] = count
    return counts_np


def furthest_first_traversal(dep_enc: Tensor, centroids: Tensor, num_clusters: int) -> Tensor:
    num_predefined = centroids.shape[0]
    # Add the centroids to the samples
    samples = torch.cat([centroids, dep_enc], dim=0)
    sampled_idxs = list(range(num_predefined))
    # Compute the euclidean distance between all pairs
    dists = get_dists(samples)
    # Mask indicating whether a sample is still yet to be sampled (1=unsampled, 0=sampled)
    # - updating a mask is far more efficient than reconstructing the list of unsampled indexes
    # every iteration (however, we do have to be careful about the 'meta-indexing' it introduces)
    unsampled_m = samples.new_ones(samples.size(0), dtype=torch.bool)
    # Mark the predefined centroids as visited
    unsampled_m[sampled_idxs] = 0

    # Begin the furthest-first traversal algorithm
    while len(sampled_idxs) < num_clusters:
        # p := argmax min_{i\inB}(d(x, x_i)); i.e. select the point which maximizes the minimum
        # squared Euclidean-distance to all previously selected points
        # NOTE: The argmax index is relative to the unsampled indexes
        max_idx_rel = torch.argmax(torch.min(dists[~unsampled_m][:, unsampled_m], dim=0).values)
        # Retrieve the absolute index corresponding to the previously-computed argmax index
        max_idx_abs = unsampled_m.nonzero()[max_idx_rel]
        # Update the mask, which corresponds to moving the sampled index from the unsampled pool to
        # the sampled pool
        unsampled_m[max_idx_abs] = 0
        # Append it to the list as well to avoid having to recompute it later
        sampled_idxs.append(int(max_idx_abs))

    return samples[sampled_idxs]


# @torch.jit.script
def get_dists(embeddings: Tensor) -> Tensor:
    print("pairwise difference...")
    # res = faiss.StandardGpuResources()
    # n = embeddings.shape[0]
    # out = np.empty((n, n), dtype=np.float32)
    # dist = torch.from_numpy(
    #     faiss.pairwise_distance_gpu(res, embeddings.numpy(), embeddings.numpy(), out, faiss.METRIC_INNER_PRODUCT)
    # )
    dist = torch.from_numpy(
        faiss.pairwise_distances(
            embeddings.numpy(),
            embeddings.numpy(),  # mt=faiss.METRIC_INNER_PRODUCT
        )
    )
    # dist = torch.from_numpy(squareform(pdist(embeddings.numpy())))
    # embeddings = embeddings.to("cuda:0")
    # dist = F.pdist(embeddings).cpu()
    print("done.")
    return dist
    # dist_mat = embeddings @ embeddings.t()
    # sq = dist_mat.diagonal().view(embeddings.size(0), 1)
    # return -2 * dist_mat + sq + sq.t()


if __name__ == "__main__":
    main()
