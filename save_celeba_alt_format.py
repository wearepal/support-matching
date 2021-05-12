from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
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
    train_inds = datasets.train.indices
    context_inds = datasets.context.indices
    # train_inds = torch.cat([train_inds, context_inds], dim=0)
    test_inds = datasets.test.indices

    assert isinstance(cfg, SaveDataConfig)

    filename = (
        f"{cfg.data.log_name}_seed={cfg.data.data_split_seed}_bias={cfg.bias.log_dataset}.csv"
    )

    img_ids_tr = datasets.train.x[train_inds]
    s_tr = datasets.train.s[train_inds]
    y_tr = datasets.train.y[train_inds]

    with open(filename, "w") as f:
        for i, s, y in zip(img_ids_tr.unbind(0), s_tr.unbind(0), y_tr.unbind(0)):
            print("%s %d %d" % (i, s, y), file=f)

    # attr_file_path = Path(cfg.data.root) / "celeba" / "list_attr_celeba.txt"
    # img = np.genfromtxt(attr_file_path, skip_header=2, dtype=str, usecols=0)
    # attr_file = pd.read_csv(attr_file_path)
    # # sens_male = np.genfromtxt(
    # #     cfg.path_to_sens, skip_header=2, dtype=int, usecols=21
    # # )  # 21 Male/Female
    # smiling = np.genfromtxt(cfg.path_to_target, skip_header=2, dtype=int, usecols=32)


if __name__ == "__main__":
    main()
