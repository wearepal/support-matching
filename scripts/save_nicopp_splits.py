import platform
from sys import argv

from conduit.data.datasets.vision import NICOPP
import numpy as np
import torch
import wandb

from src.data.common import find_data_dir
from src.data.splitter import save_split_inds_as_artifact


def main(seed: int) -> None:
    assert seed >= 0
    run = wandb.init(
        project="support-matching", entity="predictive-analytics-lab", dir="local_logging"
    )
    NICOPP.data_split_seed = seed
    ds = NICOPP(root=find_data_dir())
    split_ids = ds.metadata["split"]
    train_inds = torch.as_tensor(np.nonzero(split_ids == NICOPP.Split.TRAIN.value)[0])
    test_inds = torch.as_tensor(np.nonzero(split_ids == NICOPP.Split.TEST.value)[0])
    dep_inds = torch.as_tensor(np.nonzero(split_ids == NICOPP.Split.VAL.value)[0])
    name_of_machine = platform.node()
    save_split_inds_as_artifact(
        run=run,
        train_inds=train_inds,
        test_inds=test_inds,
        dep_inds=dep_inds,
        ds=ds,
        seed=seed,
        artifact_name=f"split_nicopp_change_is_hard_{name_of_machine}_{seed}",
    )
    run.finish()


if __name__ == "__main__":
    main(int(argv[1]))
