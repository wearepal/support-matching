from collections.abc import Iterable
from dataclasses import dataclass, field

from loguru import logger
import numpy as np
import numpy.typing as npt
from ranzen.torch import batchwise_pdist
from sklearn.cluster import KMeans as SklKMeans
from sklearn.cluster import kmeans_plusplus
import torch
from torch import Tensor

from src.data import DataModule, resolve_device

from .encode import Encodings
from .metrics import evaluate

__all__ = ["KMeans", "fft_init"]


def fft_init(
    x: Tensor,
    *,
    centroids: Tensor,
    num_clusters: int,
    device: str | torch.device | None = None,
    chunk_size: int = 1000,
) -> Tensor:
    num_predefined = len(centroids)
    # Add the centroids to the samples
    xc = torch.cat([centroids, x], dim=0)
    sampled_idxs = list(range(num_predefined))
    # Compute the euclidean distance between all pairs
    logger.info("Computing pairwise differences...")
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        # on the GPU, we have to do it batch-wise
        dists = batchwise_pdist(xc.to(device), chunk_size=chunk_size).cpu()
    else:
        dists = torch.cdist(x1=xc, x2=xc)
    # Mask indicating whether a sample is still yet to be sampled (1=unsampled, 0=sampled)
    # - updating a mask is far more efficient than reconstructing the list of unsampled indexes
    # every iteration (however, we do have to be careful about the 'meta-indexing' it introduces)
    unsampled_m = xc.new_ones((len(xc),), dtype=torch.bool)
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

    return xc[sampled_idxs]


def precompute_centroids(
    train_enc: Tensor, *, train_group_ids: Tensor, all_group_ids: Iterable[int]
) -> Tensor:
    """Determine centroids from group IDs."""
    centroids: list[Tensor] = []
    for group in all_group_ids:
        mask = train_group_ids == group
        if mask.count_nonzero() > 0:
            centroids.append(train_enc[mask].mean(0, keepdim=True))
    return torch.cat(centroids, dim=0)


@dataclass
class KMeans:
    n_init: int = 10
    fft_cluster_init: bool = False
    supervised_cluster_init: bool = False
    spherical: bool = True
    device: int | str | torch.device = 0
    _fitted_model: SklKMeans | None = field(
        default=None, init=False, metadata={"omegaconf_ignore": True}
    )

    def fit(self, dm: DataModule, *, encodings: Encodings) -> None:
        device = resolve_device(self.device)
        if self.spherical:
            # l2-normalize the encodings so Euclidean distance is converted into cosine distance.
            encodings.normalize_(p=2)
        train_clusters = dm.group_ids_tr.unique()
        dep_clusters = dm.group_ids_dep.unique()
        n_clusters = len(dep_clusters)
        logger.info(f"clusters in deployment set: {n_clusters}; in training: {len(train_clusters)}")

        if self.supervised_cluster_init:
            logger.info("Using pre-computed centroids")
            centroids: Tensor = precompute_centroids(
                encodings.train,
                train_group_ids=encodings.train_labels,
                all_group_ids=train_clusters,
            )
            if len(centroids) < n_clusters:
                logger.info(f"Need additional clusters: {n_clusters - len(centroids)}")
                if self.fft_cluster_init:
                    logger.info(
                        "Using furthest-first traversal to generate additional initial clusters..."
                    )
                    centroids_np = fft_init(
                        encodings.dep, centroids=centroids, num_clusters=n_clusters, device=device
                    ).numpy()
                else:
                    logger.info("Using kmeans++ to generate additional initial clusters...")
                    additional_centroids, _ = kmeans_plusplus(
                        encodings.dep.numpy(),
                        n_clusters=n_clusters - len(centroids),
                        random_state=0,
                    )
                    centroids_np = np.concatenate([centroids.numpy(), additional_centroids], axis=0)
                logger.info("Done.")
            else:
                centroids_np = centroids.numpy()
            kmeans = SklKMeans(n_clusters=n_clusters, init=centroids_np, n_init=1)
        else:
            logger.info("Using kmeans++")
            kmeans = SklKMeans(n_clusters=n_clusters, init="k-means++", n_init=self.n_init)
        kmeans.fit(encodings.to_cluster)
        preds = kmeans.predict(encodings.test)
        evaluate(y_true=encodings.test_labels, y_pred=preds, use_wandb=True, prefix="clustering")

        self._fitted_model = kmeans

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        if self._fitted_model is None:
            raise AttributeError(
                "Can't generate predictions as model has not yet been fitted; please call 'fit' "
                "first."
            )
        return self._fitted_model.predict(data)
