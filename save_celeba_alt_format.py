from dataclasses import dataclass
from pathlib import Path
import random
from typing import Type, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, to_absolute_path
import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING, OmegaConf
import pandas as pd
import torch

from shared.configs.arguments import (
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

    save_dir: str = ""
    seed: int = 0

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="save_data_schema", node=SaveDataConfig)
register_configs()


@hydra.main(config_path="conf", config_name="data_gen")
def main(hydra_config: DictConfig) -> None:
    cfg = OmegaConf.to_object(hydra_config)
    assert isinstance(cfg, SaveDataConfig)
    if not isinstance(cfg.data, CelebaConfig):
        raise ValueError("Data-saving currently only works for CelebA.")
    #  Load the datasets and wrap with dataloaders

    np.random.seed(cfg.seed)  # cpu vars
    torch.manual_seed(cfg.seed)  # cpu  vars
    random.seed(cfg.seed)  # Python

    datasets = load_dataset(cfg)  # type: ignore

    assert isinstance(cfg, SaveDataConfig)

    base_filename = f"{cfg.data.log_name}_seed={cfg.seed}_bias={cfg.bias.log_dataset}"

    for split in ("train", "context", "test"):
        subset = getattr(datasets, split)
        split_inds = subset.indices

        img_ids_tr = subset.dataset.x[split_inds]
        s_tr = subset.dataset.s[split_inds]
        y_tr = subset.dataset.y[split_inds]

        save_dir = Path(to_absolute_path(cfg.save_dir))
        save_dir.mkdir(exist_ok=True, parents=True)
        split_filename = save_dir / Path(f"{base_filename}_{split}.txt")
        with open(split_filename, "w") as f:
            for i, s, y in zip(img_ids_tr, s_tr, y_tr):
                print("%s %d %d" % (i, s, y), file=f)
        print(f"{split} data saved to {split_filename}")


if __name__ == "__main__":
    main()
