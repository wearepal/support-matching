from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any, Dict, Final, List, Optional, Union, cast

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets import CdtVisionDataset
from conduit.data.datasets.utils import ImageTform, PillowTform, stratified_split
from loguru import logger
from pandas.io.formats.style_render import TypedDict
from ranzen import implements
import torch
from torch import Tensor
import torchvision.transforms as T
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from .common import D, Dataset, TrainDepTestSplit

__all__ = [
    "DataSplitter",
    "RandomSplitter",
    "SplitFromArtifact",
    "load_split_inds_from_artifact",
    "save_split_inds_as_artifact",
]


@dataclass(eq=False)
class DataSplitter:
    transductive: bool = False  # whether to include the test data in the pool of unlabelled data
    train_transforms: Any = None
    test_transforms: Any = None
    dep_transforms: Any = None

    @classmethod
    def _default_train_transforms(cls) -> ImageTform:
        transform_ls: list[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    @classmethod
    def _default_test_transforms(cls) -> ImageTform:
        transform_ls: List[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    @abstractmethod
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        ...

    def __call__(self, dataset: D) -> TrainDepTestSplit[D]:
        splits = self.split(dataset)
        # Enable transductive learning (i.e. using the test data for semi-supervised learning)
        if self.transductive:
            splits.deployment = splits.deployment.cat(splits.test, inplace=False)
        # Assign transforms if datasets are vision ones
        if isinstance(splits.train, CdtVisionDataset):
            splits.train.transform = (
                self._default_train_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
        if isinstance(splits.deployment, CdtVisionDataset):
            splits.deployment.transform = (
                splits.train.transform if self.dep_transforms is None else self.dep_transforms  # type: ignore
            )
        if isinstance(splits.test, CdtVisionDataset):
            splits.test.transform = self.test_transforms
            splits.test.transform = (
                self._default_test_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )

        return splits


FILENAME: Final[str] = "split_inds.pt"


class SavedSplitInds(TypedDict):
    train: Tensor
    dep: Tensor
    test: Tensor


def save_split_inds_as_artifact(
    run: Optional[Union[Run, RunDisabled]],
    *,
    train_inds: Tensor,
    test_inds: Tensor,
    dep_inds: Tensor,
    ds: Dataset,
    seed: int,
    artifact_name: Optional[str] = None,
) -> Optional[str]:
    if run is None:
        run = cast(Optional[Run], wandb.run)
        if run is None:
            logger.info(
                f"No active wandb run with which to save an artifact: skipping saving of splits."
            )
            return None
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ds_str = str(ds.__class__.__name__).lower()
        # Store the name of machine (as reported by operating system) as the seed is
        # machine-dependent.
        name_of_machine = platform.node()
        metadata = {"dataset": ds_str, "seed": seed, "machine": name_of_machine}
        if artifact_name is None:
            artifact_name = f"split_{ds_str}_{name_of_machine}_{seed}"
        save_path = tmpdir / FILENAME
        to_save = {"train": train_inds, "dep": dep_inds, "test": test_inds}
        torch.save(to_save, f=save_path)
        artifact = wandb.Artifact(artifact_name, type="split_inds", metadata=metadata)
        artifact.add_file(str(save_path.resolve()), name=FILENAME)
        run.log_artifact(artifact)
        artifact.wait()
    versioned_name = f"{run.entity}/{run.project}/{artifact_name}:{artifact.version}"
    logger.info(f"Split indices saved to {versioned_name}")
    return versioned_name


@dataclass(eq=False)
class RandomSplitter(DataSplitter):
    seed: int = 42
    dep_prop: float = 0.4
    test_prop: float = 0.2
    # The propotion of the dataset to use overall (pre-splitting)
    data_prop: float = 1.0

    # Dataset manipulation
    dep_subsampling_props: Optional[Dict[int, Any]] = None
    train_subsampling_props: Optional[Dict[int, Any]] = None
    artifact_name: Optional[str] = None
    save_as_artifact: bool = False

    def __post_init__(self) -> None:
        if not (0 < self.data_prop <= 1):
            raise AttributeError("'data_prop' must be in the range (0, 1].")
        if (self.data_prop < 1) and self.save_as_artifact:
            raise AttributeError("'data_prop' < 1 is incompatible with 'save_as_artifact'.")
        if not (0 < self.test_prop < 1):
            raise AttributeError("'test_prop' must be in the range (0, 1).")
        if not (0 < self.dep_prop < 1):
            raise AttributeError("'dep_prop' must be in the range (0, 1).")

    @implements(DataSplitter)
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        if self.data_prop < 1:
            dataset = stratified_split(dataset, default_train_prop=self.data_prop).train
        dep_inds, test_inds, train_inds = dataset.random_split(
            props=[self.dep_prop, self.test_prop],
            seed=self.seed,
            as_indices=True,
        )
        train_inds = torch.as_tensor(train_inds)
        train_data = dataset.subset(train_inds)
        if self.train_subsampling_props is not None:
            logger.info(
                "Subsampling training set with proportions:\n\t"
                f"{str(self.train_subsampling_props)}"
            )
            train_inds_ss = torch.as_tensor(
                stratified_split(
                    train_data,
                    default_train_prop=1.0,
                    train_props=self.train_subsampling_props,
                    seed=self.seed,
                    as_indices=True,
                ).train
            )
            train_data = train_data.subset(train_inds_ss)
            train_inds = train_inds[train_inds_ss]

        dep_inds = torch.as_tensor(dep_inds)
        dep_data = dataset.subset(dep_inds)
        if self.dep_subsampling_props is not None:
            logger.info(
                "Subsampling deployment set with proportions:\n\t"
                f"{str(self.dep_subsampling_props)}"
            )
            dep_inds_ss = torch.as_tensor(
                stratified_split(
                    dep_data,
                    default_train_prop=1.0,
                    train_props=self.dep_subsampling_props,
                    as_indices=True,
                    seed=self.seed,
                ).train
            )
            dep_data = dep_data.subset(dep_inds_ss)
            dep_inds = dep_inds[dep_inds_ss]

        test_data = dataset.subset(test_inds)

        if self.save_as_artifact:
            save_split_inds_as_artifact(
                run=wandb.run,
                train_inds=train_inds,
                dep_inds=dep_inds,
                test_inds=torch.as_tensor(test_inds),
                ds=dataset,
                seed=self.seed,
                artifact_name=self.artifact_name,
            )

        return TrainDepTestSplit(train=train_data, deployment=dep_data, test=test_data)


def _process_root_dir(root: Optional[Union[Path, str]]) -> Path:
    if root is None:
        root = Path("artifacts", "splits")
    elif isinstance(root, str):
        root = Path(root)
    return root


def load_split_inds_from_artifact(
    run: Optional[Union[Run, RunDisabled]],
    *,
    name: str,
    ds: Dataset,
    project: Optional[str] = None,
    root: Optional[Union[Path, str]] = None,
    version: Optional[int] = None,
) -> SavedSplitInds:
    root = _process_root_dir(root)
    version_str = ":latest" if version is None else f":v{version}"
    artifact_dir = root / name / version_str
    versioned_name = name + version_str
    filepath = artifact_dir / FILENAME
    if not filepath.exists():
        if (run is not None) and (project is None):
            project = f"{run.entity}/{run.project}"
            full_name = f"{project}/{versioned_name}"
            artifact = run.use_artifact(full_name)
            logger.info("Downloading split-indices artifact...")
            artifact.download(root=artifact_dir)
        else:
            raise RuntimeError(
                f"No pre-existing artifact found at location '{filepath.resolve()}'"
                "and because no wandb run has been specified, it can't be downloaded."
            )
    full_name = artifact_dir
    split_inds = torch.load(filepath)
    logger.info(f"Split indices successfully loaded from artifact '{full_name}'.")
    return split_inds


@dataclass(eq=False)
class _ArtifactLoaderMixin:
    artifact_name: str
    version: Optional[int] = None


@dataclass(eq=False)
class SplitFromArtifact(DataSplitter, _ArtifactLoaderMixin):
    @implements(DataSplitter)
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        splits = load_split_inds_from_artifact(
            run=wandb.run, name=self.artifact_name, version=self.version, ds=dataset
        )
        train_data = dataset.subset(splits["train"])
        dep_data = dataset.subset(splits["dep"])
        test_data = dataset.subset(splits["test"])
        return TrainDepTestSplit(train=train_data, deployment=dep_data, test=test_data)
