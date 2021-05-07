from __future__ import annotations
from abc import abstractmethod
import logging
from pathlib import Path

from torch.tensor import Tensor
import wandb
import yaml

from shared.configs.arguments import CmnistConfig, Config
from shared.data.data_loading import DatasetTriplet, load_dataset
from shared.utils.utils import as_pretty_dict, flatten_dict, random_seed
from shared.utils.utils import random_seed

__all__ = ["AlgBase"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class AlgBase:
    """Base class for algorithms."""

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.misc_cfg = cfg.misc
        self.bias_cfg = cfg.bias

    def _to_device(self, *tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Place tensors on the correct device."""
        moved = [tensor.to(self.misc_cfg.device, non_blocking=True) for tensor in tensors]

        return moved[0] if len(moved) == 1 else tuple(moved)

    @abstractmethod
    def _fit(self, datasets: DatasetTriplet) -> AlgBase:
        ...

    def run(self, cluster_label_file: Path | None = None) -> None:
        """Main function.

        Args:
            hydra_config: configuration object from hydra
            cluster_label_file: path to a pth file with cluster IDs

        Returns:
            the trained encoder
        """

        random_seed(self.misc_cfg.seed, self.misc_cfg.use_gpu)

        run = None
        if self.misc_cfg.use_wandb:
            project_suffix = (
                f"-{self.data_cfg.log_name}" if not isinstance(self.data_cfg, CmnistConfig) else ""
            )
            group = ""
            if self.misc_cfg.log_method:
                group += self.misc_cfg.log_method
            if self.misc_cfg.exp_group:
                group += "." + self.misc_cfg.exp_group
            if self.bias_cfg.log_dataset:
                group += "." + self.bias_cfg.log_dataset
            local_dir = Path(".", "local_logging")
            local_dir.mkdir(exist_ok=True)
            run = wandb.init(
                entity="predictive-analytics-lab",
                project="suds-hydra" + project_suffix,
                dir=str(local_dir),
                config=flatten_dict(as_pretty_dict(self.cfg)),
                group=group if group else None,
                reinit=True,
            )
            run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

        LOGGER.info(
            yaml.dump(
                as_pretty_dict(self.cfg),
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=True,
            )
        )

        # ==== construct dataset ====
        datasets: DatasetTriplet = load_dataset(self.cfg)
        LOGGER.info(
            "Size of context-set: {}, training-set: {}, test-set: {}".format(
                len(datasets.context),  # type: ignore
                len(datasets.train),  # type: ignore
                len(datasets.test),  # type: ignore
            )
        )

        self._fit(datasets=datasets)

        if run is not None:
            run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
