from dataclasses import dataclass
from typing import TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING, OmegaConf
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

T = TypeVar("T", bound="BaseConfig")


@dataclass
class SaveDataConfig:
    """Minimum config needed to do data loading."""

    save_dir: str = ""

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="save_data_schema", node=SaveDataConfig)
register_configs()


@hydra.main(config_path="conf", config_name="data_gen")
def main(hydra_config: DictConfig) -> None:
    cfg: object = OmegaConf.to_object(hydra_config)
    assert isinstance(cfg, SaveDataConfig)
    if not isinstance(cfg.data, CelebaConfig):
        raise ValueError("Data-saving currently only works for CelebA.")
    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)
    train_inds = datasets.train.indices
    context_inds = datasets.context.indices
    train_inds = torch.cat([train_inds, context_inds], dim=0)
    test_inds = datasets.test.indices

    split_inds = [train_inds, train_inds, test_inds]
    # train_inds is intentionally repeated as GEORGE expects the data to be split into
    # train/val/test sets but in our setting we have no notion of the second of these
    all_inds = torch.cat([train_inds, train_inds, test_inds], dim=0).to_numpy()
    split_inds = torch.cat(
        [torch.full_like(inds, split_ind) for split_ind, inds in enumerate(split_inds)], dim=0
    ).to_numpy()
    split_df = pd.DataFrame({"indices": all_inds, "partition": split_inds})
    filename = (
        f"{cfg.data.log_name}_seed={cfg.data.data_split_seed}_bias={cfg.bias.log_dataset}.csv"
    )
    split_df.to_csv(filename, sep=r"\s")
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()
