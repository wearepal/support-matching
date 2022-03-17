from __future__ import annotations
import logging
import platform
from typing import NamedTuple

from conduit.data.datasets.utils import stratified_split
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset

from shared.configs import BaseConfig

__all__ = ["DataModule", "load_dataset"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class DataModule(NamedTuple):
    context: Dataset
    test: Dataset
    train: Dataset
    s_dim: int
    y_dim: int


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    data_dirs = {
        "m900382.inf.susx.ac.uk": "/Users/tk324/PycharmProjects/NoSINN/data",
        "turing": "/srv/galene0/shared/data",
        "fear": "/srv/galene0/shared/data",
        "hydra": "/srv/galene0/shared/data",
        "goedel": "/srv/galene0/shared/data",
    }
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return data_dirs.get(name_of_machine, to_absolute_path("data"))


def load_dataset(cfg: BaseConfig) -> DataModule:
    args = cfg.data
    root = args.root or find_data_dir()
    all_data = instantiate(args, root=root, split=None)

    # elif isinstance(args, AdultConfig):
    #     context_data, train_data, test_data = load_adult_data(cfg)
    #     y_dim = 1
    #     if args.adult_split is AdultDatasetSplit.Education:
    #         s_dim = 3
    #     elif args.adult_split is AdultDatasetSplit.Sex:
    #         s_dim = 1
    #     else:
    #         raise ValueError(f"This split is not yet fully supported: {args.adult_split}")
    # else:
    #     raise ValueError(f"Invalid choice of dataset: {args}")

    # if 0 < args.data_pcnt < 1:
    #     context_data = shrink_dataset(context_data, args.data_pcnt)
    #     train_data = shrink_dataset(train_data, args.data_pcnt)
    #     test_data = shrink_dataset(test_data, args.data_pcnt)
    # # Enable transductive learning (i.e. using the test data for semi-supervised learning)
    # if cfg.misc.cache_data:
    #     LOGGER.info("Caching all three datasets...")
    #     context_data = _cache_data(context_data)
    #     test_data = _cache_data(test_data)
    #     train_data = _cache_data(train_data)
    #     LOGGER.info("Done.")

    context_data, test_data, train_data = all_data.random_split(
        props=[args.context_pcnt, args.test_pcnt]
    )
    train_data = stratified_split(
        train_data, default_train_prop=1.0, train_props=cfg.bias.subsample_train
    ).train

    if cfg.bias.subsample_context:
        LOGGER.info("Subsampling context set...")
        context_data = stratified_split(
            train_data, default_train_prop=1.0, train_props=cfg.bias.subsample_context
        ).train

    if args.transductive:
        context_data = ConcatDataset([context_data, test_data])

    y_dim = train_data.dim_y[0]
    s_dim = train_data.dim_s[0]

    return DataModule(
        context=context_data,
        test=test_data,
        train=train_data,
        s_dim=s_dim,
        y_dim=y_dim,
    )
