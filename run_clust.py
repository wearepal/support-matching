"""Simply call the main function."""
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from clustering.optimisation import main
from shared.configs import Config

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: Config):
    cfg: Config = instantiate(hydra_config, _convert_="partial")
    main(cfg=cfg)


if __name__ == "__main__":
    app()
