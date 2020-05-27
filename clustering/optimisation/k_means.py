import time
from typing import Tuple

import numpy as np
from pykeops.torch import LazyTensor
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from shared.utils import wandb_log
from clustering.configs import ClusterArgs
from clustering.models import Encoder
from .evaluation import encode_dataset
from .utils import count_occurances, find_assignment


def train(
    args: ClusterArgs, encoder: Encoder, context_data: Dataset, num_clusters: int, s_count: int
) -> TensorDataset:
    # encode the training set with the encoder
    encoded = encode_dataset(args, context_data, encoder)
    # create data loader with one giant batch
    data_loader = DataLoader(encoded, batch_size=len(encoded), shuffle=False)
    encoded, s, y = next(iter(data_loader))
    preds, _ = k_means(encoded, num_clusters, device=args._device, n_iter=args.epochs, verbose=True)
    counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    counts = count_occurances(counts, preds.cpu().numpy(), s, y, s_count, args.cluster)
    _, logging_dict = find_assignment(counts, preds.size(0))
    prepared = (
        f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in logging_dict.items()
    )
    print(" | ".join(prepared))
    wandb_log(args, logging_dict, step=0)
    return TensorDataset(encoded, preds)


def k_means(
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

    return cl, c
