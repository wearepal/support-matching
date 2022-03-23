from collections import defaultdict
import logging
from pathlib import Path
import platform
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)
from typing_extensions import Self, TypeAlias

import albumentations as A
import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets import CdtDataset, CdtVisionDataset
from conduit.data.datasets.utils import (
    CdtDataLoader,
    ImageTform,
    PillowTform,
    get_group_ids,
    stratified_split,
)
from conduit.data.structures import LoadedData, MeanStd, TernarySample
from conduit.logging import init_logger
from conduit.transforms import denormalize
from hydra.utils import instantiate, to_absolute_path
from omegaconf.dictconfig import DictConfig
from ranzen.torch.data import (
    BaseSampler,
    BatchSamplerBase,
    SequentialBatchSampler,
    StratifiedBatchSampler,
    TrainingMode,
)
import torch
from torch import Tensor
import torchvision.transforms.transforms as T

from shared.configs.arguments import DataModuleConf, SplitConf
from shared.utils.loadsave import ClusterResults
from shared.utils.utils import lcm

from .utils import group_id_to_label

__all__ = [
    "D",
    "DataModule",
    "Dataset",
    "TrainContextTestSplit",
]


Dataset: TypeAlias = CdtDataset[TernarySample[LoadedData], Any, Tensor, Tensor]
D = TypeVar("D", bound=Dataset)


@attr.define(kw_only=True)
class TrainContextTestSplit(Generic[D]):
    train: D
    context: D
    test: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.context, self.test)

    def num_samples(self) -> int:
        return len(self.train) + len(self.context) + len(self.test)

    @property
    def num_train_samples(self) -> int:
        return len(self.train)

    @property
    def num_context_samples(self) -> int:
        return len(self.context)

    @property
    def num_test_samples(self) -> int:
        return len(self.test)


@attr.define(kw_only=True)
class DataModule(Generic[D]):
    LOGGER: ClassVar[logging.Logger] = init_logger("MissingSourceDataModule")

    DATA_DIRS: ClassVar[Dict[str, str]] = {
        "m900382.inf.susx.ac.uk": "/Users/tk324/PycharmProjects/NoSINN/data",
        "turing": "/srv/galene0/shared/data",
        "fear": "/srv/galene0/shared/data",
        "hydra": "/srv/galene0/shared/data",
        "goedel": "/srv/galene0/shared/data",
    }

    train: D
    context: D
    test: D

    # DataLoader settings
    batch_size_tr: int
    _batch_size_ctx: Optional[int] = None
    _batch_size_te: Optional[int] = None
    num_samples_per_group_per_bag: int = 1

    num_workers: int = 4
    persist_workers: bool = False
    pin_memory: bool = True

    balanced_context: bool = False

    @property
    def batch_size_ctx(self) -> int:
        return self.batch_size_tr if self._batch_size_ctx is None else self._batch_size_ctx

    @batch_size_ctx.setter
    def batch_size_ctx(self, value: Optional[int]) -> None:
        self._batch_size_ctx = value

    @property
    def batch_size_te(self) -> int:
        return self.batch_size_tr if self.batch_size_te is None else self.batch_size_te

    @batch_size_te.setter
    def batch_size_te(self, value: Optional[int]) -> None:
        self._batch_size_te = value

    @classmethod
    def find_data_dir(cls: Type[Self]) -> str:
        """Find data directory for the current machine based on predefined mappings."""
        name_of_machine = platform.node()  # name of machine as reported by operating system
        return cls.DATA_DIRS.get(name_of_machine, to_absolute_path("data"))

    @property
    def num_train_samples(self) -> int:
        return len(self.train)

    @property
    def num_context_samples(self) -> int:
        return len(self.context)

    @property
    def num_test_samples(self) -> int:
        return len(self.test)

    @property
    def dim_x(self) -> torch.Size:
        return self.train.dim_x

    @property
    def dim_s(self) -> int:
        return self.train.dim_s[0]

    @property
    def dim_y(self) -> int:
        return self.train.dim_y[0]

    @property
    def card_y(self) -> int:
        return self.train.card_y

    @property
    def card_s(self) -> int:
        return self.train.card_s

    @property
    def num_classes(self) -> int:
        return max(2, self.card_y)

    @property
    def bag_size(self) -> int:
        return self.card_y * self.card_s * self.num_samples_per_group_per_bag

    @property
    def feature_group_slices(self) -> Optional[Dict[str, List[slice]]]:
        return None

    @classmethod
    def _default_train_transforms(cls) -> ImageTform:
        transform_ls: List[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    @classmethod
    def _default_test_transforms(cls) -> ImageTform:
        transform_ls: List[PillowTform] = []
        transform_ls.append(T.ToTensor())
        transform_ls.append(T.Normalize(mean=IMAGENET_STATS.mean, std=IMAGENET_STATS.std))
        return T.Compose(transform_ls)

    def _make_dataloader(
        self,
        ds: D,
        *,
        batch_size: Optional[int],
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[BatchSamplerBase] = None,
    ) -> CdtDataLoader[TernarySample]:
        """Make DataLoader."""
        return CdtDataLoader(
            ds,
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
        )

    @staticmethod
    def _get_multipliers(group_ids: Tensor, *, card_s: int) -> Dict[int, int]:
        """This is a standalone function only because then we can have a unit test for it."""
        unique_ids = group_ids.unique(sorted=False).tolist()

        # first, count how many subgroups there are for each y
        num_subgroups_per_y: DefaultDict[int, int] = defaultdict(int)
        for group_id in unique_ids:
            corresponding_y = group_id_to_label(group_id=group_id, s_count=card_s, label="y")
            num_subgroups_per_y[corresponding_y] += 1

        # To make all subgroups effectively the same size, we first scale everything by the least common
        # multiplier and then we divide by the number of subgroups to get the final multiplier.
        largest_multiplier = lcm(num_subgroups_per_y.values())
        multipliers = {}
        for group_id in unique_ids:
            num_subgroups = num_subgroups_per_y[
                group_id_to_label(group_id=group_id, s_count=card_s, label="y")
            ]
            multiplier = largest_multiplier // num_subgroups
            multipliers[group_id] = multiplier

        return multipliers

    def _make_stratified_sampler(
        self, group_ids: Tensor, *, batch_size: int
    ) -> StratifiedBatchSampler:
        multipliers = self._get_multipliers(group_ids, card_s=self.test.card_s)
        num_samples_per_group = self.num_samples_per_group_per_bag * batch_size
        return StratifiedBatchSampler(
            group_ids=group_ids.squeeze().tolist(),
            num_samples_per_group=num_samples_per_group,
            shuffle=True,
            base_sampler=BaseSampler.sequential,
            multipliers=multipliers,
            training_mode=TrainingMode.step,
            drop_last=False,
        )

    def train_dataloader(
        self, eval: bool = False, *, balance: bool = True, batch_size: Optional[int] = None
    ) -> CdtDataLoader[TernarySample]:
        if eval:
            return self._make_dataloader(
                ds=self.train, batch_size=self.batch_size_te, shuffle=False
            )
        if balance:
            group_ids = get_group_ids(self.train)
            batch_sampler = self._make_stratified_sampler(
                group_ids=group_ids, batch_size=self.batch_size_tr
            )
            batch_size = None
        else:
            batch_size = self.batch_size_tr if batch_size is None else batch_size
            batch_sampler = None
        return self._make_dataloader(
            ds=self.train,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )

    def context_dataloader(
        self,
        cluster_results: Optional[ClusterResults] = None,
        eval: bool = False,
    ) -> CdtDataLoader[TernarySample]:
        if eval:
            return self._make_dataloader(
                ds=self.context, batch_size=self.batch_size_te, shuffle=False
            )

        group_ids: Optional[Tensor] = None
        if self.balanced_context:
            group_ids = get_group_ids(self.context)
        elif cluster_results is not None:
            group_ids = cluster_results.class_ids
        if group_ids is None:
            batch_sampler = SequentialBatchSampler(
                data_source=self.context,
                batch_size=self.batch_size_ctx,
                shuffle=True,
                training_mode=TrainingMode.step,
                drop_last=False,
            )
        else:
            batch_sampler = self._make_stratified_sampler(
                group_ids=group_ids,
                batch_size=self.batch_size_ctx,
            )
        return self._make_dataloader(
            ds=self.context, batch_size=1, batch_sampler=batch_sampler
        )

    def test_dataloader(self) -> CdtDataLoader[TernarySample]:
        return self._make_dataloader(ds=self.test, batch_size=self.batch_size_te, shuffle=False)

    @classmethod
    def _generate_splits(
        cls: Type[Self], dataset: D, split_config: SplitConf
    ) -> TrainContextTestSplit[D]:

        context_data, test_data, train_data = dataset.random_split(
            props=[split_config.context_prop, split_config.test_prop]
        )

        cls.LOGGER.info("Subsampling training set...")
        train_data = stratified_split(
            train_data,
            default_train_prop=1.0,
            train_props=split_config.subsample_train,
            seed=split_config.seed,
        ).train

        if split_config.subsample_context:
            cls.LOGGER.info("Subsampling context set...")
            context_data = stratified_split(
                train_data,
                default_train_prop=1.0,
                train_props=split_config.subsample_context,
                seed=split_config.seed,
            ).train

        # Enable transductive learning (i.e. using the test data for semi-supervised learning)
        if split_config.transductive:
            context_data = context_data.cat(test_data, inplace=False)

        # Assign transforms if datasets are vision ones
        if isinstance(train_data, CdtVisionDataset):
            train_data.transform = (
                cls._default_train_transforms()
                if split_config.train_transforms is None
                else split_config.train_transforms
            )
        if isinstance(context_data, CdtVisionDataset):
            context_data.transform = split_config.context_transforms
            train_data = cast(CdtVisionDataset, train_data)
            context_data.transform = (
                train_data.transform
                if split_config.context_transforms is None
                else split_config.context_transforms
            )
        if isinstance(test_data, CdtVisionDataset):
            test_data.transform = split_config.test_transforms
            test_data.transform = (
                cls._default_test_transforms()
                if split_config.test_transforms is None
                else split_config.test_transforms
            )

        return TrainContextTestSplit(train=train_data, context=context_data, test=test_data)

    @classmethod
    def from_configs(
        cls: Type[Self],
        *,
        dm_config: DataModuleConf,
        ds_config: DictConfig,
        split_config: SplitConf,
    ) -> Self:
        if ds_config.root is None:
            root = cls.find_data_dir()
        else:
            root = str(Path(to_absolute_path(ds_config.root)).resolve())
        all_data: D = instantiate(ds_config, root=root, split=None)
        if split_config.data_prop is not None:
            all_data = stratified_split(all_data, default_train_prop=split_config.data_prop).train
        splits = cls._generate_splits(dataset=all_data, split_config=split_config)
        return cls(
            train=splits.train,
            context=splits.context,
            test=splits.test,
            **dm_config,  # type: ignore
        )

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.context, self.test)

    def __str__(self) -> str:
        ds_name = self.train.__class__.__name__
        size_info = (
            f"- Size of training-set: {self.num_train_samples}\n"
            f"- Size of context-set: {self.num_context_samples}\n"
            f"- Size of test-set: {self.num_test_samples}"
        )
        return f"\nDataModule for dataset of type {ds_name}\n{size_info}"

    def denormalize(self, x: Tensor, *, inplace: bool = False) -> Tensor:
        if isinstance(self.train, CdtVisionDataset):
            if (tform := self.train.transform) is not None:

                def _get_stats(_tform: ImageTform) -> Optional[MeanStd]:
                    stats = None
                    if isinstance(_tform, (T.Normalize, A.Normalize)):
                        stats = MeanStd(mean=_tform.mean, std=_tform.std)

                    elif isinstance(_tform, (T.Compose, A.Compose)):
                        inner_tforms = list(_tform.transforms)
                        for inner_tform in inner_tforms[::-1]:
                            stats = _get_stats(inner_tform)
                            if stats is not None:
                                break
                    return stats

                if (stats := _get_stats(tform)) is not None:
                    return denormalize(x, mean=stats.mean, std=stats.std, inplace=inplace)
        return x
