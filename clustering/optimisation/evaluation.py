import logging
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from clustering.models import BaseModel, Encoder
from shared.configs import Config

__all__ = ["classify_dataset", "encode_dataset"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


def encode_dataset(
    cfg: Config, data: Dataset, generator: Encoder
) -> "Dataset[Tuple[Tensor, Tensor, Tensor]]":
    LOGGER.info("Encoding dataset...")
    all_enc = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=cfg.clust.encode_batch_size, pin_memory=True, shuffle=False, num_workers=0
    )

    device = torch.device(cfg.misc.device)
    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            enc = generator.encode(x, stochastic=False)
            all_enc.append(enc.detach().cpu())

    all_enc = torch.cat(all_enc, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_enc, all_s, all_y)
    LOGGER.info("Done.")

    return encoded_dataset


def classify_dataset(cfg: Config, model: BaseModel, data: Dataset) -> Tuple[Tensor, Tensor, Tensor]:
    """Determine the class of every sample in the given dataset and save them to a file."""
    model.eval()
    cluster_ids: List[Tensor] = []
    all_s: List[Tensor] = []
    all_y: List[Tensor] = []

    data_loader = DataLoader(
        data, batch_size=cfg.clust.encode_batch_size, pin_memory=True, shuffle=False, num_workers=0
    )

    device = torch.device(cfg.misc.device)
    with torch.set_grad_enabled(False):
        for (x, s, y) in data_loader:
            x = x.to(device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)
            logits, _ = model(x)
            preds = logits.argmax(dim=-1).detach().cpu()
            cluster_ids.append(preds)

    return torch.cat(cluster_ids, dim=0), torch.cat(all_s, dim=0), torch.cat(all_y, dim=0)
