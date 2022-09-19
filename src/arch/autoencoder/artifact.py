from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Final, Optional, Tuple, Union

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from ranzen.decorators import implements
import torch
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from src.arch.autoencoder.base import AePair

from .base import AeFactory

__all__ = [
    "AeFromArtifact",
    "load_ae_from_artifact",
    "save_ae_artifact",
]

FILENAME: Final[str] = "model.pt"


@torch.no_grad()
def save_ae_artifact(
    model: AePair,
    *,
    run: Union[Run, RunDisabled],
    config: Union[DictConfig, Dict[str, Any]],
    name: str,
) -> None:
    if isinstance(config, DictConfig):
        config = dict(config)
    assert "_target_" in config
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_save_path = tmpdir / FILENAME
        save_dict = {
            "state": model.state_dict(),
            "config": config,
        }
        torch.save(save_dict, f=model_save_path)
        logger.info(f"Model config and state saved to '{model_save_path.resolve()}'")
        model_artifact = wandb.Artifact(name, type="model", metadata=dict(config))
        model_artifact.add_file(str(model_save_path.resolve()), name=FILENAME)
        run.log_artifact(model_artifact)
        model_artifact.wait()
        logger.info(
            "Model artifact saved to "
            f"'{run.entity}/{run.project}/{name}:{model_artifact.version}'"
        )


def _process_root_dir(root: Optional[Union[Path, str]]) -> Path:
    if root is None:
        root = Path("artifacts", "autoencoder")
    elif isinstance(root, str):
        root = Path(root)
    return root


@torch.no_grad()
def load_ae_from_artifact(
    name: str,
    *,
    input_shape: Tuple[int, int, int],
    version: Optional[int] = None,
    run: Optional[Union[Run, RunDisabled]] = None,
    project: Optional[str] = None,
    root: Optional[Union[Path, str]] = None,
) -> AePair:
    root = _process_root_dir(root)
    version_str = "latest" if version is None else f"v{version}"
    versioned_name = name + f":{version_str}"
    artifact_dir = root / name / version_str
    filepath = artifact_dir / FILENAME
    if not filepath.exists():
        if run is None:
            run = wandb.run
        if (run is not None) and (project is None):
            project = f"{run.entity}/{run.project}"
            full_name = f"{project}/{versioned_name}"
            artifact = run.use_artifact(full_name)
            logger.info("Downloading model artifact...")
            artifact.download(root=artifact_dir)
        else:
            raise RuntimeError(
                f"No pre-existing model-artifact found at location '{filepath.resolve()}'"
                "and because no wandb run has been specified, it can't be downloaded."
            )
    full_name = artifact_dir
    state_dict = torch.load(filepath)
    logger.info("Loading saved parameters and buffers...")
    factory: AeFactory = instantiate(state_dict["config"])
    if isinstance(factory, AeFromArtifact):
        raise RuntimeError(
            "Cannot load in AeFactory as an artifact as this would result in infinite " "recursion."
        )
    ae_pair = factory(input_shape=input_shape)
    ae_pair.load_state_dict(state_dict["state"])
    logger.info(f"Model successfully loaded from artifact '{full_name}'.")
    return ae_pair


@dataclass(eq=False)
class AeFromArtifact(AeFactory):
    artifact_name: str
    version: Optional[int] = None
    bitfit: bool = False

    @implements(AeFactory)
    def __call__(
        self,
        input_shape: Tuple[int, int, int],
    ) -> AePair:
        ae_pair = load_ae_from_artifact(
            input_shape=input_shape, name=self.artifact_name, version=self.version
        )
        if self.bitfit:
            for name, param in ae_pair.named_parameters():
                if "bias" not in name:
                    param.requires_grad_(False)
        return ae_pair
