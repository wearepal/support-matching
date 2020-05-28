import time
from typing import Tuple, Union

import faiss
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from clustering.configs import ClusterArgs
from clustering.models import Encoder
from shared.utils import wandb_log

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
    preds = run_kmeans_faiss(
        encoded,
        nmb_clusters=num_clusters,
        cuda=args._device != "cpu",
        n_iter=args.epochs,
        verbose=True,
    )
    counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    counts = count_occurances(counts, preds, s, y, s_count, args.cluster)
    _, logging_dict = find_assignment(counts, preds.size(0))
    prepared = (
        f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in logging_dict.items()
    )
    print(" | ".join(prepared))
    wandb_log(args, logging_dict, step=0)
    return TensorDataset(encoded, preds)


def run_kmeans_faiss(
    x: Union[np.ndarray, Tensor], nmb_clusters: int, n_iter: int, cuda: bool, verbose: bool = False
) -> np.ndarray:

    if isinstance(x, torch.Tensor):
        x = x.numpy()

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

    I = np.squeeze(I)

    return I
