"""Simply call the main function."""
from clustering.optimisation import main
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from shared.configs import Config, register_configs

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: DictConfig):
    cfg = Config.from_hydra(hydra_config=hydra_config)
    main(cfg=cfg)


if __name__ == "__main__":
    app()
