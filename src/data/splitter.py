from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any, Final, TypedDict, cast
from typing_extensions import override

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets import random_split
from conduit.data.datasets.utils import stratified_split
from conduit.data.datasets.vision import CdtVisionDataset, ImageTform, PillowTform
from conduit.fair.data.datasets import ACSDataset
from conduit.transforms import MinMaxNormalize, TabularNormalize, ZScoreNormalize
from loguru import logger
from ranzen import some
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
    "TabularSplitter",
    "load_split_inds_from_artifact",
    "save_split_inds_as_artifact",
]


@dataclass(eq=False)
class DataSplitter(ABC):
    @abstractmethod
    def __call__(self, dataset: D) -> TrainDepTestSplit[D]:
        """Split the dataset into train/deployment/test."""


@dataclass(eq=False)
class _VisionDataSplitter(DataSplitter):
    """Common methods for transforming vision datasets."""

    transductive: bool = False
    """Whether to include the test data in the pool of unlabelled data."""

    train_transforms: Any = None  # T.Compose
    test_transforms: Any = None  # T.Compose
    dep_transforms: Any = None  # T.Compose

    @classmethod
    def _default_train_transforms(cls) -> ImageTform:
        transform_ls: list[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    @classmethod
    def _default_test_transforms(cls) -> ImageTform:
        transform_ls: list[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    @abstractmethod
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        raise NotImplementedError()

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
    run: Run | RunDisabled | None,
    *,
    train_inds: Tensor,
    test_inds: Tensor,
    dep_inds: Tensor,
    ds: Dataset,
    seed: int,
    artifact_name: str | None = None,
) -> str | None:
    if run is None:
        run = cast(Run | None, wandb.run)
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
class RandomSplitter(_VisionDataSplitter):
    seed: int = 42
    dep_prop: float = 0.4
    test_prop: float = 0.2
    # The propotion of the dataset to use overall (pre-splitting)
    data_prop: float = 1.0

    # Dataset manipulation
    dep_subsampling_props: dict[int, Any] | None = None
    train_subsampling_props: dict[int, Any] | None = None
    artifact_name: str | None = None
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

    @override
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        if self.data_prop < 1:
            dataset = stratified_split(dataset, default_train_prop=self.data_prop).train
        dep_inds, test_inds, train_inds = random_split(
            dataset, props=[self.dep_prop, self.test_prop], seed=self.seed, as_indices=True
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


def _process_root_dir(root: Path | str | None) -> Path:
    if root is None:
        root = Path("artifacts", "splits")
    elif isinstance(root, str):
        root = Path(root)
    return root


def load_split_inds_from_artifact(
    run: Run | RunDisabled | None,
    *,
    name: str,
    project: str | None = None,
    root: Path | str | None = None,
    version: int | None = None,
) -> SavedSplitInds:
    root = _process_root_dir(root)
    version_str = ":latest" if version is None else f":v{version}"
    artifact_dir = root / name / version_str
    versioned_name = name + version_str
    filepath = artifact_dir / FILENAME
    if not filepath.exists():
        if run is None:
            run = wandb.run
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


@dataclass(eq=False, kw_only=True)
class SplitFromArtifact(_VisionDataSplitter):
    artifact_name: str
    version: int | None = None

    @override
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        splits = load_split_inds_from_artifact(
            run=wandb.run, name=self.artifact_name, version=self.version
        )
        train_data = dataset.subset(splits["train"])
        dep_data = dataset.subset(splits["dep"])
        test_data = dataset.subset(splits["test"])
        return TrainDepTestSplit(train=train_data, deployment=dep_data, test=test_data)


class TabularTform(Enum):
    zscore_normalize = (ZScoreNormalize,)
    minmax_normalize = (MinMaxNormalize,)

    def __init__(self, tform: Callable[[], TabularNormalize]) -> None:
        self.tf = tform


@dataclass(eq=False)
class TabularSplitter(DataSplitter):
    """Split and transform tabular datasets."""

    seed: int
    train_props: dict[int, dict[int, float]] | None = None
    dep_prop: float = 0.2
    test_prop: float = 0.1
    transform: TabularTform | None = TabularTform.zscore_normalize

    @override
    def __call__(self, dataset: D) -> TrainDepTestSplit[D]:
        if not isinstance(dataset, ACSDataset):
            raise NotImplementedError("TabularSplitter only supports splitting of `ACSDataset`.")

        train, dep, test = dataset.subsampled_split(
            train_props=self.train_props,
            val_prop=self.dep_prop,
            test_prop=self.test_prop,
            seed=self.seed,
        )
        if some(tf_type := self.transform):
            tf = tf_type.tf()
            train.fit_transform_(tf)
            dep.transform_(tf)
            test.transform_(tf)

        return TrainDepTestSplit(train=train, deployment=dep, test=test)
