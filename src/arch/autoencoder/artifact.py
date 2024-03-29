from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final
from typing_extensions import override

from hydra.utils import instantiate
from loguru import logger
import torch
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from src.arch.autoencoder.base import AePair

from .base import AeFactory

__all__ = ["AeFromArtifact", "load_ae_from_artifact", "save_ae_artifact"]

FILENAME: Final[str] = "model.pt"


@torch.no_grad()  # pyright: ignore
def save_ae_artifact(
    model: AePair, *, run: Run | RunDisabled, factory_config: dict[str, Any], name: str
) -> None:
    assert "_target_" in factory_config
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_save_path = tmpdir / FILENAME
        save_dict = {
            "state": model.state_dict(),
            "config": factory_config,
        }
        torch.save(save_dict, f=model_save_path)
        logger.info(f"Model config and state saved to '{model_save_path.resolve()}'")
        model_artifact = wandb.Artifact(name, type="model", metadata=dict(factory_config))
        model_artifact.add_file(str(model_save_path.resolve()), name=FILENAME)
        run.log_artifact(model_artifact)
        model_artifact.wait()
        logger.info(
            "Model artifact saved to "
            f"'{run.entity}/{run.project}/{name}:{model_artifact.version}'"
        )


def _process_root_dir(root: Path | str | None) -> Path:
    if root is None:
        root = Path("artifacts", "autoencoder")
    elif isinstance(root, str):
        root = Path(root)
    return root


@torch.no_grad()  # pyright: ignore
def load_ae_from_artifact(
    name: str,
    *,
    input_shape: tuple[int, int, int],
    version: int | None = None,
    run: Run | RunDisabled | None = None,
    project: str | None = None,
    root: Path | str | None = None,
) -> tuple[AePair, dict[str, Any]]:
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
            "Cannot load in AeFactory as an artifact as this would result in infinite recursion."
        )
    ae_pair = factory(input_shape=input_shape)
    ae_pair.load_state_dict(state_dict["state"])
    logger.info(f"Model successfully loaded from artifact '{full_name}'.")
    return ae_pair, state_dict["config"]


@dataclass(eq=False)
class AeFromArtifact(AeFactory):
    artifact_name: str
    version: int | None = None
    bitfit: bool = False
    factory_config: dict[str, Any] = field(init=False, metadata={"omegaconf_ignore": True})

    @override
    def __call__(self, input_shape: tuple[int, int, int]) -> AePair:
        ae_pair, self.factory_config = load_ae_from_artifact(
            input_shape=input_shape, name=self.artifact_name, version=self.version
        )
        if self.bitfit:
            for name, param in ae_pair.named_parameters():
                if "bias" not in name:
                    param.requires_grad_(False)
        return ae_pair
