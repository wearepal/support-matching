from typing import Any, ClassVar, Optional, Protocol, cast

from attrs import NOTHING, Attribute, fields
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data.common import process_data_dir
from src.relay.supmatch import SupMatchRelay

__all__ = ["Experiment", "launch"]


class Experiment(Protocol):
    options: ClassVar[dict[str, dict[str, type]]]

    def run(self, raw_config: Optional[dict[str, Any]] = None) -> Optional[float]:
        ...


def launch(relay_cls: type[Experiment]) -> None:
    check_hydra_config(relay_cls, relay_cls.options)

    @hydra.main(config_path="../../external_confs", config_name="config", version_base=None)
    def main(hydra_config: DictConfig) -> Optional[float]:
        # TODO: this should probably be done somewhere else
        # Deal with missing `root`
        if OmegaConf.is_missing(hydra_config["ds"], "root"):
            hydra_config["ds"]["root"] = process_data_dir(None)
        else:
            hydra_config["ds"]["root"] = process_data_dir(hydra_config["ds"]["root"])

        # instantiate everything that has `_target_` defined
        relay = instantiate(hydra_config, _convert_="object")
        assert isinstance(relay, relay_cls)

        raw_config = OmegaConf.to_container(
            hydra_config, throw_on_missing=True, enum_to_str=False, resolve=True
        )
        assert isinstance(raw_config, dict)
        raw_config = cast(dict[str, Any], raw_config)
        raw_config = {
            f"{key}/{OmegaConf.get_type(dict_).__name__}"  # type: ignore
            if isinstance(dict_ := hydra_config[key], DictConfig)
            else key: value
            for key, value in raw_config.items()
        }
        return relay.run(raw_config)

    main()


def check_hydra_config(main_cls: type, options: dict[str, dict[str, type]]) -> None:
    """Check the given config and store everything in the ConfigStore.

    This function performs two tasks: 1) make the necessary calls to `ConfigStore`
    and 2) run some checks over the given config and if there are problems, try to give a nice
    error message.

    :param main_cls: The main config class.
    :param options: A dictionary that defines all the variants. The keys of top level of the
        dictionary should corresponds to the group names, and the keys in the nested dictionaries
        should correspond to the names of the options.
    """
    configs: tuple[Attribute, ...] = fields(main_cls)
    for config in configs:
        if config.type == Any or (isinstance(typ := config.type, str) and typ == "Any"):
            if config.name not in options:
                raise ValueError(
                    f"if an entry has type Any, there should be variants: {config.name}"
                )
            if config.default is not NOTHING:
                raise ValueError(
                    f"if an entry has type Any, there should be no default value: {config.name}"
                )
        else:
            if config.name in options:
                raise ValueError(
                    f"if an entry has a real type, there should be no variants: {config.name}"
                )
            if config.default is NOTHING:
                raise ValueError(
                    f"if an entry has a real type, there should be a default value: {config.name}"
                )

    cs = ConfigStore.instance()
    cs.store(node=main_cls, name="config_schema")
    for group, entries in options.items():
        for name, node in entries.items():
            try:
                cs.store(node=node, name=name, group=group)
            except Exception as exc:
                raise RuntimeError(f"{main_cls=}, {node=}, {name=}, {group=}") from exc


if __name__ == "__main__":
    launch(SupMatchRelay)
