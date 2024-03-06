from typing import Any, ClassVar, Protocol, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ranzen.hydra import register_hydra_config

from src.data.common import process_data_dir
from src.relay.supmatch import SupMatchRelay

__all__ = ["Experiment", "launch"]


class Experiment(Protocol):
    options: ClassVar[dict[str, dict[str, type]]]

    def run(self, raw_config: dict[str, Any] | None = None) -> float | None: ...


def launch(relay_cls: type[Experiment]) -> None:
    register_hydra_config(relay_cls, relay_cls.options, schema_name="config_schema")

    @hydra.main(config_path="../../conf", config_name="config", version_base=None)
    def main(hydra_config: DictConfig) -> float | None:
        if "root" in hydra_config["ds"]:
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


if __name__ == "__main__":
    launch(SupMatchRelay)
