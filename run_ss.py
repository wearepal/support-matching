"""Simply call the main function."""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from advrep.algs import AdvSemiSupervisedAlg, LAFTR, SupportMatching
from shared.configs import AdaptationMethod, Config, register_configs

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
register_configs()


def main(cfg: Config) -> None:
    alg: AdvSemiSupervisedAlg
    if cfg.adapt.method is AdaptationMethod.suds:
        alg = SupportMatching(cfg=cfg)
    else:
        alg = LAFTR(cfg=cfg)
    alg.run()


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: DictConfig):
    cfg = Config.from_hydra(hydra_config=hydra_config)
    main(cfg)


if __name__ == "__main__":
    app()
