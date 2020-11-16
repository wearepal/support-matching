"""Simply call the main function."""
import hydra

from clustering.optimisation import main
from shared.configs import Config

@hydra.main(config_path="conf", config_name="config")
def app(cfg: Config):
    main(cfg=cfg)

if __name__ == "__main__":
    app()
