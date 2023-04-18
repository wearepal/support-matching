import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.relay import SupMatchRelay
from src.relay.base import fix_dataset_root

cs = ConfigStore.instance()
cs.store(node=SupMatchRelay, name="config_schema")
for group, entries in SupMatchRelay.options.items():
    for name, node in entries.items():
        cs.store(node=node, name=name, group=group)


@hydra.main(config_path="external_confs", config_name="config", version_base="1.2")
def main(hydra_config: DictConfig) -> None:
    fix_dataset_root(hydra_config)
    # instantiate everything that has `_target_` defined
    omega_dict = instantiate(hydra_config)
    relay = OmegaConf.to_object(omega_dict)
    assert isinstance(relay, SupMatchRelay)
    relay.run(
        OmegaConf.to_container(hydra_config, throw_on_missing=True, enum_to_str=False, resolve=True)
    )


if __name__ == "__main__":
    main()
