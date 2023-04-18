from typing import Any, ClassVar, Optional, Protocol

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.data.common import process_data_dir

__all__ = ["Experiment", "generic_run"]


class Experiment(Protocol):
    options: ClassVar[dict[str, dict[str, type]]]

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> Optional[float]:
        ...


def generic_run(relay_cls: type[Experiment]) -> None:
    cs = ConfigStore.instance()
    cs.store(node=relay_cls, name="config_schema")
    for group, entries in relay_cls.options.items():
        for name, node in entries.items():
            cs.store(node=node, name=name, group=group)

    @hydra.main(config_path="external_confs", config_name="config", version_base="1.2")
    def main(hydra_config: DictConfig) -> None:
        # TODO: this should probably be done somewhere else
        # Deal with missing `root`
        if OmegaConf.is_missing(hydra_config["ds"], "root"):
            hydra_config["ds"]["root"] = process_data_dir(None)
        else:
            hydra_config["ds"]["root"] = process_data_dir(hydra_config["ds"]["root"])

        # instantiate everything that has `_target_` defined
        relay = instantiate(hydra_config, _convert_="object")
        assert isinstance(relay, relay_cls)

        relay.run(
            OmegaConf.to_container(
                hydra_config, throw_on_missing=True, enum_to_str=False, resolve=True
            )
        )

    main()
