from typing import Any, ClassVar, Optional, Protocol

from attrs import NOTHING, Attribute, fields
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data.common import process_data_dir
from src.relay.supmatch import SupMatchRelay

__all__ = ["Experiment", "run"]


class Experiment(Protocol):
    options: ClassVar[dict[str, dict[str, type]]]

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> Optional[float]:
        ...


def run(relay_cls: type[Experiment]) -> None:
    # verify some aspects of the configs
    configs: tuple[Attribute, ...] = fields(relay_cls)
    for config in configs:
        if config.type == Any or (isinstance(typ := config.type, str) and typ == "Any"):
            if config.name not in relay_cls.options:
                raise ValueError(
                    f"if an entry has type Any, there should be variants: {config.name}"
                )
            if config.default is not NOTHING:
                raise ValueError(
                    f"if an entry has type Any, there should be no default value: {config.name}"
                )
        else:
            if config.name in relay_cls.options:
                raise ValueError(
                    f"if an entry has a real type, there should be no variants: {config.name}"
                )
            if config.default is NOTHING:
                raise ValueError(
                    f"if an entry has a real type, there should be a default value: {config.name}"
                )

    cs = ConfigStore.instance()
    cs.store(node=relay_cls, name="config_schema")
    for group, entries in relay_cls.options.items():
        for name, node in entries.items():
            try:
                cs.store(node=node, name=name, group=group)
            except Exception as exc:
                raise RuntimeError(f"{relay_cls=}, {node=}, {name=}, {group=}") from exc

    @hydra.main(config_path="external_confs", config_name="config", version_base=None)
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


if __name__ == "__main__":
    run(SupMatchRelay)
