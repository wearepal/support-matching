from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets import CdtVisionDataset
from conduit.data.datasets.utils import ImageTform, PillowTform, stratified_split
from loguru import logger
from ranzen import implements
import torchvision.transforms as T

from .common import D, TrainDepTestSplit

__all__ = [
    "DataSplitter",
    "RandomSplitter",
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
            splits.deployment.transform = self.dep_transforms
            splits.train = cast(CdtVisionDataset, splits.train)
            splits.train.transform = (
                splits.train.transform if self.dep_transforms is None else self.dep_transforms
            )
        if isinstance(splits.test, CdtVisionDataset):
            splits.test.transform = self.test_transforms
            splits.test.transform = (
                self._default_test_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )

        return splits


@dataclass(eq=True)
class RandomSplitter(DataSplitter):
    seed: int = 42
    dep_prop: float = 0.4
    test_prop: float = 0.2
    # The propotion of the dataset to use overall (pre-splitting)
    data_prop: Optional[float] = None

    # Dataset manipulation
    dep_subsampling_props: Optional[Dict[int, Any]] = None
    train_subsampling_props: Optional[Dict[int, Any]] = None

    @implements(DataSplitter)
    def split(self, dataset: D) -> TrainDepTestSplit[D]:
        if self.data_prop is not None:
            dataset = stratified_split(dataset, default_train_prop=self.data_prop).train
        dep_data, test_data, train_data = dataset.random_split(
            props=[self.dep_prop, self.test_prop],
            seed=self.seed,
        )
        logger.info(
            "Subsampling training set with proportions:\n\t" f"{str(self.train_subsampling_props)}"
        )
        train_data = stratified_split(
            train_data,
            default_train_prop=1.0,
            train_props=self.train_subsampling_props,
            seed=self.seed,
        ).train

        if self.dep_subsampling_props:
            logger.info("Subsampling deployment set...")
            dep_data = stratified_split(
                dep_data,
                default_train_prop=1.0,
                train_props=self.dep_subsampling_props,
                seed=self.seed,
            ).train

        return TrainDepTestSplit(train=train_data, deployment=dep_data, test=test_data)
