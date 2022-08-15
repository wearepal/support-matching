from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans as _KMeans
from sklearn.cluster import kmeans_plusplus
import torch
from torch import Tensor
from tqdm import tqdm

from shared.data import DataModule

from .encode import Encodings
from .metrics import evaluate

__all__ = [
    "KMeans",
    "batchwise_pdist",
    "fft_init",
]

LOGGER = logging.getLogger(__file__)


def fft_init(x: Tensor, *, centroids: Tensor, num_clusters: int, use_gpu: bool = True) -> Tensor:
    num_predefined = centroids.shape[0]
    # Add the centroids to the samples
    xc = torch.cat([centroids, x], dim=0)
    sampled_idxs = list(range(num_predefined))
    # Compute the euclidean distance between all pairs
    LOGGER.info("pairwise difference...")
    if use_gpu:
        # on the GPU, we have to do it batch-wise
        dists = batchwise_pdist(xc.to("cuda:0"), chunk_size=1000).cpu()
    else:
        dists = torch.cdist(x1=xc, x2=xc)
    # Mask indicating whether a sample is still yet to be sampled (1=unsampled, 0=sampled)
    # - updating a mask is far more efficient than reconstructing the list of unsampled indexes
    # every iteration (however, we do have to be careful about the 'meta-indexing' it introduces)
    unsampled_m = xc.new_ones(len(xc), dtype=torch.bool)
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


def batchwise_pdist(x: Tensor, *, chunk_size: int, p_norm: float = 2.0) -> Tensor:
    """Compute pdist in batches.

    This is necessary because if you compute pdist directly, it doesn't fit into memory.
    """
    chunks = torch.split(x, split_size_or_sections=chunk_size)

    columns: list[Tensor] = []
    for chunk in tqdm(chunks):
        shards = [torch.cdist(chunk, other_chunk, p_norm) for other_chunk in chunks]
        column = torch.cat(shards, dim=1)
        # the result has to be moved to the CPU; otherwise we'll run out of GPU memory
        columns.append(column.cpu())

        # free up memory
        for shard in shards:
            del shard
        del column
        torch.cuda.empty_cache()

    dists = torch.cat(columns, dim=0)

    # free up memory
    for column in columns:
        del column
    torch.cuda.empty_cache()
    return dists


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
    use_fft: bool = False
    use_labels: bool = False
    _fitted_model: Optional[_KMeans] = None

    def fit(self, dm: DataModule, *, enc: Encodings) -> None:
        train_clusters = dm.group_ids_tr.unique()
        dep_clusters = dm.group_ids_dep.unique()
        n_clusters = len(dep_clusters)

        if self.use_labels:
            LOGGER.info("Using pre-computed centroids")
            centroids: Tensor = precompute_centroids(
                enc.train, train_group_ids=enc.train_labels, all_group_ids=train_clusters
            )
            if len(centroids) < n_clusters:
                LOGGER.info(f"Need additional clusters: {n_clusters - len(centroids)}")
                if self.use_fft:
                    LOGGER.info(
                        "Using furthest-first traversal to generate additional initial clusters..."
                    )
                    centroids_np = fft_init(
                        enc.dep, centroids=centroids, num_clusters=n_clusters
                    ).numpy()
                else:
                    LOGGER.info("Using kmeans++ to generate additional initial clusters...")
                    additional_centroids, _ = kmeans_plusplus(
                        enc.dep.numpy(), n_clusters=n_clusters - len(centroids), random_state=0
                    )
                    centroids_np = np.concatenate([centroids.numpy(), additional_centroids], axis=0)
                LOGGER.info("Done.")
            else:
                centroids_np = centroids.numpy()
            kmeans = _KMeans(n_clusters=n_clusters, init=centroids_np, n_init=1)
            kmeans.fit(enc.to_cluster)
            preds = kmeans.predict(enc.test)
            evaluate(y_true=enc.test_labels, y_pred=preds)
        else:
            LOGGER.info("Using kmeans++")
            kmeans = _KMeans(n_clusters=n_clusters, init="k-means++", n_init=self.n_init)
            kmeans.fit(enc.to_cluster)
            preds = kmeans.predict(enc.test)
            evaluate(y_true=enc.test_labels, y_pred=preds)

        self._fitted_model = kmeans

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        if self._fitted_model is None:
            raise AttributeError(
                "Can't generate predictions as model has not yet been fitted; please call 'fit' "
                "first."
            )
        return self._fitted_model.predict(data)
