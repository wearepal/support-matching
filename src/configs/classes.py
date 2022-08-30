from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DataModuleConf",
]


@dataclass
class DataModuleConf:
    batch_size_tr: int = 1
    batch_size_te: Optional[int] = None
    num_samples_per_group_per_bag: int = 1
    num_workers: int = 0
    persist_workers: bool = False
    pin_memory: bool = True
    gt_deployment: bool = True
    # Amount of noise to apply to the labels used for balanced sampling
    # -- only applicable when ``gt_deployment=True``
    label_noise: float = 0.0
    seed: int = 47
