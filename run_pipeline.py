"""Call the main functions of both parts one after the other."""
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from shared.configs import Config, register_configs

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: DictConfig) -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    cfg = Config.from_hydra(hydra_config=hydra_config)

    with TemporaryDirectory() as tmpdir:
        clf = Path(tmpdir) / "labels.pth"

        from clustering.optimisation import main as cluster

        cluster(cfg=cfg, cluster_label_file=clf)

        from suds.optimisation import main as disentangle

        disentangle(cfg=cfg, cluster_label_file=clf)


if __name__ == "__main__":
    app()
