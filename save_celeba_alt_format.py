from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, to_absolute_path
import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING
import pandas as pd
import torch

from shared.configs.arguments import (
    BaseConfig,
    BiasConfig,
    CelebaConfig,
    DatasetConfig,
    register_configs,
)
from shared.data.data_loading import load_dataset


T = TypeVar("T", bound="SaveDataConfig")


@dataclass
class SaveDataConfig:
    """Minimum config needed to do data loading."""

    _target_: str = "SaveDataConfig"
    save_dir: str = ""

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING

    @classmethod
    def from_hydra(cls: Type[T], hydra_config: DictConfig) -> T:
        """Instantiate this class based on a hydra config.

        This is necessary because dataclasses cannot be instantiated recursively yet.
        """
        subconfigs = {
            k: instantiate(v) for k, v in hydra_config.items() if k not in ("_target_", "save_dir")
        }

        return cls(**subconfigs)


cs = ConfigStore.instance()
cs.store(name="save_data_schema", node=SaveDataConfig)
register_configs()


@hydra.main(config_path="conf", config_name="data_gen")
def main(hydra_config: DictConfig) -> None:
    cfg = SaveDataConfig.from_hydra(hydra_config)
    if not isinstance(cfg.data, CelebaConfig):
        raise ValueError("Data-saving currently only works for CelebA.")
    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)

    assert isinstance(cfg, SaveDataConfig)

    base_filename = (
        f"{cfg.data.log_name}_seed={cfg.data.data_split_seed}_bias={cfg.bias.log_dataset}.csv"
    )

    for split in ("train", "context", "test"):
        subset = getattr(datasets, split)
        split_inds = subset.indices

        img_ids_tr = subset.dataset.x[split_inds]
        s_tr = subset.dataset.s[split_inds]
        y_tr = subset.dataset.y[split_inds]

        split_filename = to_absolute_path(cfg.save_dir) / Path(f"{base_filename}_{split}")
        with open(split_filename, "w") as f:
            for i, s, y in zip(img_ids_tr, s_tr, y_tr):
                print("%s %d %d" % (i, s, y), file=f)
        print(f"{split} data saved to {split_filename.resolve()}")


if __name__ == "__main__":
    main()
