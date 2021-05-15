"""Simply call the main function."""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from shared.configs import Config, register_configs
from shared.configs.enums import AdaptationMethod
from suds.algs.laftr import LAFTR
from suds.algs.supmatch import SupportMatching

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: DictConfig):
    cfg = Config.from_hydra(hydra_config=hydra_config)
    if cfg.adapt.method is AdaptationMethod.suds:
        alg = SupportMatching(cfg=cfg)
    else:
        alg = LAFTR(cfg=cfg)
    alg.run()


if __name__ == "__main__":
    app()
