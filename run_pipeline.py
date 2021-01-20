"""Call the main functions of both parts one after the other."""
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from shared.configs import Config

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def app(hydra_config: Config) -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    with TemporaryDirectory() as tmpdir:
        clf = Path(tmpdir) / "labels.pth"

        from clustering.optimisation import main as cluster

        cluster(cfg=hydra_config, cluster_label_file=clf)

        from fdm.optimisation import main as disentangle

        cfg: Config = instantiate(hydra_config, _convert_="partial")
        disentangle(cfg=cfg, cluster_label_file=clf)


if __name__ == "__main__":
    app()
