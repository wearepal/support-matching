"""Simply call the main function."""
import hydra
from hydra.core.config_store import ConfigStore

from clustering.optimisation import main
from shared.configs import Config

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def app(cfg: Config):
    main(cfg=cfg)


if __name__ == "__main__":
    app()
