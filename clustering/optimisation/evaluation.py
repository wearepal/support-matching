from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from clustering.configs import ClusterArgs
from clustering.models import Encoder, Model

from .utils import log_images

__all__ = ["classify_dataset", "encode_dataset"]


def log_sample_images(args: ClusterArgs, data: Dataset, name: str, step: int) -> None:
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(args, x, f"Samples from {name}", prefix="eval", step=step)


def encode_dataset(
    args: ClusterArgs, data: Dataset, generator: Encoder
) -> "Dataset[Tuple[Tensor, Tensor, Tensor]]":
    print("Encoding dataset...", flush=True)  # flush to avoid conflict with tqdm
    all_enc = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=args.encode_batch_size, pin_memory=True, shuffle=False, num_workers=4
    )

    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(args._device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            # if the following line throws an error, it's Myles' fault
            enc = generator.encode(x, stochastic=False)
            all_enc.append(enc.detach().cpu())

    all_enc = torch.cat(all_enc, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_enc, all_s, all_y)
    print("Done.")

    return encoded_dataset


def classify_dataset(
    args: ClusterArgs, model: Model, data: Dataset
) -> Tuple[Tensor, Tensor, Tensor]:
    """Determine the class of every sample in the given dataset and save them to a file."""
    model.eval()
    cluster_ids: List[Tensor] = []
    all_s: List[Tensor] = []
    all_y: List[Tensor] = []

    data_loader = DataLoader(
        data, batch_size=args.encode_batch_size, pin_memory=True, shuffle=False, num_workers=4
    )

    with torch.set_grad_enabled(False):
        for (x, s, y) in data_loader:
            x = x.to(args._device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)
            logits = model(x)
            preds = logits.argmax(dim=-1).detach().cpu()
            cluster_ids.append(preds)

    return torch.cat(cluster_ids, dim=0), torch.cat(all_s, dim=0), torch.cat(all_y, dim=0)
