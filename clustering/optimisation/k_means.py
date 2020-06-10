import time
from typing import Tuple, Union

import numpy as np
from pykeops.torch import LazyTensor
import faiss
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from shared.utils import wandb_log, ClusterResults
from clustering.configs import ClusterArgs
from clustering.models import Encoder
from .evaluation import encode_dataset
from .utils import count_occurances, find_assignment, get_class_id


def train(
    args: ClusterArgs, encoder: Encoder, context_data: Dataset, num_clusters: int, s_count: int
) -> ClusterResults:
    # encode the training set with the encoder
    encoded = encode_dataset(args, context_data, encoder)
    # create data loader with one giant batch
    data_loader = DataLoader(encoded, batch_size=len(encoded), shuffle=False)
    encoded, s, y = next(iter(data_loader))
    preds = run_kmeans_faiss(
        encoded,
        nmb_clusters=num_clusters,
        cuda=str(args._device) != "cpu",
        n_iter=args.epochs,
        verbose=True,
    )
    # preds, _ = run_kmeans_torch(encoded, num_clusters, device=args._device, n_iter=args.epochs, verbose=True)
    counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    counts, _ = count_occurances(counts, preds.cpu().numpy(), s, y, s_count, args.cluster)
    context_acc, _, logging_dict = find_assignment(counts, preds.size(0))
    prepared = (
        f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in logging_dict.items()
    )
    print(" | ".join(prepared))
    wandb_log(args, logging_dict, step=0)
    return ClusterResults(
        flags=args.as_dict(),
        cluster_ids=preds,
        class_ids=get_class_id(s=s, y=y, s_count=s_count, to_cluster=args.cluster),
        context_acc=context_acc,
    )


def run_kmeans_torch(
    x: torch.Tensor, k: int, device: torch.device, n_iter: int = 10, verbose: bool = True
) -> Tuple[Tensor, Tensor]:
    x = x.flatten(start_dim=1)
    N, D = x.shape  # Number of samples, dimension of the ambient space
    dtype = torch.float64 if device.type == "cpu" else torch.float32

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:k, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    print("Finding K means...", flush=True)  # flush to avoid conflict with tqdm
    for _ in tqdm(range(n_iter)):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        # (Npoints, Nclusters) symbolic matrix of squared distances
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print(f"K-means with {N:,} points in dimension {D:,}, K = {k:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                n_iter, end - start, n_iter, (end - start) / n_iter
            )
        )

    return c, c


def run_kmeans_faiss(
    x: Union[np.ndarray, Tensor], nmb_clusters: int, n_iter: int, cuda: bool, verbose: bool = False
) -> Tensor:
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    x = np.reshape(x, (x.shape[0], -1))
    n_data, d = x.shape

    if cuda:
        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)
        clus.niter = n_iter
        clus.max_points_per_centroid = 10000000
        clus.verbose = verbose
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        flat_config.device = 0
        _, I = index.search(x, 1)
    else:
        kmeans = faiss.Kmeans(d=d, k=nmb_clusters, verbose=verbose, niter=20)
        kmeans.train(x)
        _, I = kmeans.index.search(x, 1)

    I = torch.as_tensor(I, dtype=torch.long).squeeze()

    return I
