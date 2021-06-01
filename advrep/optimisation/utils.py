from __future__ import annotations
from collections import defaultdict
import logging
from pathlib import Path
from typing import Sequence, Union, cast

from ethicml.vision.data.image_dataset import TorchImageDataset
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import ConcatDataset, Subset
import torchvision
import wandb

from shared.configs import CelebaConfig, Config, IsicConfig, ReconstructionLoss
from shared.data.dataset_wrappers import DataTupleDataset, TensorDataTupleDataset
from shared.data.isic import IsicDataset
from shared.utils import StratifiedSampler, as_pretty_dict, flatten_dict
from shared.utils.utils import class_id_to_label, label_to_class_id, lcm

__all__ = [
    "ExtractableDataset",
    "build_weighted_sampler_from_dataset",
    "extract_labels_from_dataset",
    "get_stratified_sampler",
    "get_stratified_sampler",
    "log_attention",
    "log_images",
    "restore_model",
    "save_model",
]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


def log_images(
    cfg: Config,
    images: Tensor,
    name: str,
    step: int,
    nsamples: int | Sequence[int] = 64,
    ncols: int = 8,
    monochrome: bool = False,
    prefix: str | None = None,
    caption: str | None = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"

    if cfg.enc.recon_loss is not ReconstructionLoss.ce and isinstance(
        cfg.data, (CelebaConfig, IsicConfig)
    ):
        images = 0.5 * images + 0.5

    if monochrome:
        images = images.mean(dim=1, keepdim=True)

    if isinstance(nsamples, int):
        blocks = [images[:nsamples]]
    else:
        blocks = []
        start_index = 0
        for num in nsamples:
            blocks.append(images[start_index : start_index + num])
            start_index += num

    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = [
        torchvision.utils.make_grid(block, nrow=ncols, pad_value=1.0).clamp(0, 1).cpu()
        for block in blocks
    ]
    shw = [
        wandb.Image(torchvision.transforms.functional.to_pil_image(i), caption=caption) for i in shw
    ]
    wandb.log({prefix + name: shw}, step=step)


def log_attention(
    cfg: Config,
    images: Tensor,
    attention_weights: Tensor,
    name: str,
    step: int,
    nbags: int,
    border_width: int = 3,
    ncols: int = 8,
    prefix: str | None = None,
):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"

    if cfg.enc.recon_loss == ReconstructionLoss.ce and images.ndim == 5:
        images = images.argmax(dim=1).float() / 255
    else:
        if isinstance(cfg.data, (CelebaConfig, IsicConfig)):
            images = 0.5 * images + 0.5

    images = images.view(*attention_weights.shape, *images.shape[1:])
    images = images[:nbags].cpu()
    attention_weights = attention_weights[:nbags]
    padding = attention_weights.view(nbags, -1, 1, 1, 1)

    w_padding = padding.expand(-1, -1, 3, border_width, images.size(-1)).cpu()
    images = torch.cat([w_padding, images, w_padding], dim=-2)
    h_padding = padding.expand(-1, -1, 3, images.size(-2), border_width).cpu()
    images = torch.cat([h_padding, images, h_padding], dim=-1)

    shw = [
        torchvision.utils.make_grid(block, nrow=ncols, pad_value=1.0).clamp(0, 1)
        for block in images.unbind(dim=0)
    ]
    shw = [
        wandb.Image(torchvision.transforms.functional.to_pil_image(image), caption=f"bag_{i}")
        for i, image in enumerate(shw)
    ]
    wandb.log({prefix + name: shw}, step=step)


def save_model(cfg: Config, save_dir: Path, model: nn.Module, itr: int, best: bool = False) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{itr}.pth"
    save_dict = {
        "args": flatten_dict(as_pretty_dict(cfg)),
        "model": model.state_dict(),
        "itr": itr,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(cfg: Config, filename: Path, model: nn.Module) -> tuple[nn.Module, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    assert cfg.enc.levels == args_chkpt["enc.levels"]

    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]


def get_stratified_sampler(
    group_ids: Tensor, oversample: bool, batch_size: int, min_size: int | None = None
) -> StratifiedSampler:
    unique, counts = torch.unique(group_ids, sorted=False, return_counts=True)

    n_used_clusters = counts.size(0)
    multipliers = {}
    if min_size is not None:
        smallest_used_cluster = int(counts.max())
        for cluster, count in zip(unique, counts):
            count_int = int(count)
            if count_int < min_size:
                LOGGER.info(f"Dropping cluster {cluster} with only {count_int} elements.")
                LOGGER.info("Consider setting --oversample to True (or improve clustering).")
                # set this cluster's multiplier to 0 so that it is skipped
                multipliers[int(cluster)] = 0
                n_used_clusters -= 1
            elif count_int < smallest_used_cluster:
                smallest_used_cluster = count_int
    else:
        smallest_used_cluster = int(counts.min())
    group_size = int(counts.max()) if oversample else smallest_used_cluster
    num_samples = n_used_clusters * group_size
    assert num_samples > batch_size, f"not enough training samples ({num_samples}) to fill a batch"
    return StratifiedSampler(
        group_ids.squeeze().tolist(), group_size, replacement=oversample, multipliers=multipliers
    )


_Dataset = Union[
    TensorDataTupleDataset, DataTupleDataset, Subset[Union[TorchImageDataset, IsicDataset]]
]
ExtractableDataset = Union[ConcatDataset[_Dataset], _Dataset]


def extract_labels_from_dataset(dataset: ExtractableDataset) -> tuple[Tensor, Tensor]:
    def _extract(dataset: _Dataset):
        if isinstance(dataset, Subset):
            _s = cast(Tensor, dataset.dataset.s[dataset.indices])  # type: ignore
            _y = cast(Tensor, dataset.dataset.y[dataset.indices])  # type: ignore
        elif isinstance(dataset, DataTupleDataset):
            _s = torch.as_tensor(dataset.s)
            _y = torch.as_tensor(dataset.y)
        else:
            _s = dataset.s
            _y = dataset.y
        return _s, _y

    if isinstance(dataset, ConcatDataset):
        s_all_ls, y_all_ls = [], []
        for _dataset in dataset.datasets:
            s, y = _extract(_dataset)  # type: ignore
            s_all_ls.append(s)
            y_all_ls.append(y)
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    else:
        s_all, y_all = _extract(dataset)  # type: ignore
    return s_all, y_all


def build_weighted_sampler_from_dataset(
    dataset: ExtractableDataset,
    s_count: int,
    oversample: bool,
    batch_size: int,
    balance_hierarchical: bool,
) -> StratifiedSampler:
    #  Extract the s and y labels in a dataset-agnostic way (by iterating)

    s_all, y_all = extract_labels_from_dataset(dataset=dataset)

    # Balance the batches of the training set via weighted sampling
    class_ids = label_to_class_id(s=s_all, y=y_all, s_count=s_count).view(-1)
    if balance_hierarchical:
        # Here we make sure that in a batch, y is balanced and within the y subsets, s is balanced.
        # So, if, for y=1, there are only samples with s=1, but for y=0, there's s=0 and s=1,
        # then the samples with y=1/s=1 get a multiplier of 2.
        multipliers, group_sizes = _get_multipliers_and_group_size(class_ids, s_count)
        return StratifiedSampler(
            group_ids=class_ids.tolist(),
            num_samples_per_group=max(group_sizes) if oversample else min(group_sizes),
            replacement=oversample,
            multipliers=multipliers,
        )
    else:
        return get_stratified_sampler(
            group_ids=class_ids, oversample=oversample, batch_size=batch_size
        )


def _get_multipliers_and_group_size(
    class_ids: Tensor, s_count: int
) -> tuple[dict[int, int], list[int]]:
    """This is a standalone function only because then we can have a unit test for it."""
    unique_classes, counts = torch.unique(class_ids, sorted=False, return_counts=True)
    class_ids_and_counts = [(int(i), int(c)) for i, c in zip(unique_classes, counts)]

    # first, count how many subgroups there are for each y
    num_subgroups_per_y: defaultdict[int, int] = defaultdict(int)
    for class_id, count in class_ids_and_counts:
        corresponding_y = class_id_to_label(class_id, s_count, "y")
        num_subgroups_per_y[corresponding_y] += 1

    # To make all subgroups effectively the same size, we first scale everything by the least common
    # multiplier and then we divide by the number of subgroups to get the final multiplier.
    largest_multiplier = lcm(num_subgroups_per_y.values())
    multipliers = {}
    group_sizes = []
    for class_id, count in class_ids_and_counts:
        num_subgroups = num_subgroups_per_y[class_id_to_label(class_id, s_count, "y")]
        multiplier = largest_multiplier // num_subgroups
        multipliers[class_id] = multiplier
        group_sizes.append(count // multiplier)  # for book keeping, groups need to be smaller
    return multipliers, group_sizes
