from __future__ import annotations
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Final, Optional, cast

from loguru import logger
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from src.data import DataModule

__all__ = [
    "load_labels_from_artifact",
    "save_labels_as_artifact",
]
FILENAME: Final[str] = "labels.pt"


def _artifact_info_from_dm(datamodule: DataModule) -> tuple[str, dict[str, str | int]]:
    ds_str = str(datamodule.train.__class__.__name__).lower()
    # Embed the name of machine (as reported by operating system) in the name
    # as the seed is machine-dependent.
    name_of_machine = platform.node()
    metadata = {"ds": ds_str, "seed": datamodule.seed}
    return f"{ds_str}_{datamodule.seed}_{name_of_machine}", metadata


def save_labels_as_artifact(
    run: Run | RunDisabled | None,
    *,
    labels: Tensor | npt.NDArray,
    datamodule: DataModule,
) -> Optional[str]:
    if run is None:
        run = cast(Optional[Run], wandb.run)
        if run is None:
            logger.info(
                f"No active wandb run with which to save an artifact: skippinng saving of labels."
            )
            return None
    if isinstance(labels, np.ndarray):
        labels = torch.as_tensor(labels, dtype=torch.long)
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        artifact_name, metadata = _artifact_info_from_dm(datamodule)
        save_path = tmpdir / FILENAME
        torch.save(labels, f=save_path)
        artifact = wandb.Artifact(artifact_name, type="labels", metadata=metadata)
        artifact.add_file(str(save_path.resolve()), name=FILENAME)
        run.log_artifact(artifact)
        artifact.wait()
    versioned_name = f"{run.entity}/{run.project}/{artifact_name}:{artifact.version}"
    logger.info(f"Cluster labels saved to {versioned_name}")
    return versioned_name


def _process_root_dir(root: Path | str | None) -> Path:
    if root is None:
        root = Path("artifacts", "clustering")
    elif isinstance(root, str):
        root = Path(root)
    return root


def load_labels_from_artifact(
    run: Run | RunDisabled | None,
    *,
    datamodule: DataModule,
    project: str | None = None,
    root: Path | str | None = None,
    version: int | None = None,
) -> Tensor:
    root = _process_root_dir(root)
    name, _ = _artifact_info_from_dm(datamodule)
    version_str = ":latest" if version is None else f":v{version}"
    artifact_dir = root / name / version_str
    versioned_name = name + version_str
    filepath = artifact_dir / FILENAME
    if (run is not None) and (project is None):
        project = f"{run.entity}/{run.project}"
        full_name = f"{project}/{versioned_name}"
        artifact = run.use_artifact(full_name)
        logger.info("Downloading model artifact...")
        artifact.download(root=artifact_dir)
    else:
        if not filepath.exists():
            raise RuntimeError(
                f"No pre-existing artifact found at location '{filepath.resolve()}'"
                "and because no wandb run has been specified, it can't be downloaded."
            )
        full_name = artifact_dir
    labels = torch.load(filepath)
    logger.info(f"Cluster labels successfully loaded from artifact '{full_name}'.")
    return labels
