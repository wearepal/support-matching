from collections import defaultdict
import logging
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
)
from typing_extensions import Self, TypeAlias

import albumentations as A
import attr
from conduit.data.datasets import CdtDataset, CdtVisionDataset
from conduit.data.datasets.utils import (
    CdtDataLoader,
    ImageTform,
    get_group_ids,
    stratified_split,
)
from conduit.data.structures import LoadedData, MeanStd, TernarySample
from conduit.logging import init_logger
from conduit.transforms import denormalize
from hydra.utils import instantiate, to_absolute_path
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

from shared.configs import BaseConfig
from shared.configs.arguments import SplitConfig
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
    batch_size_ctx: int = attr.field()
    batch_size_te: int = attr.field()
    num_samples_per_group_per_bag: int = 1

    num_workers: int = 4
    persist_workers: bool = False
    pin_memory: bool = True

    balanced_context: bool = False

    @batch_size_ctx.default  # type: ignore
    def _batch_size_ctx_default(self) -> int:
        return self.batch_size_tr

    @batch_size_te.default  # type: ignore
    def _batch_size_te_default(self) -> int:
        return self.batch_size_tr

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
            ds=self.context, batch_size=self.batch_size_ctx, batch_sampler=batch_sampler
        )

    def test_dataloader(self) -> CdtDataLoader[TernarySample]:
        return self._make_dataloader(ds=self.test, batch_size=self.batch_size_te, shuffle=False)

    @classmethod
    def _generate_splits(
        cls: Type[Self], dataset: D, split_config: SplitConfig
    ) -> TrainContextTestSplit[D]:

        context_data, test_data, train_data = dataset.random_split(
            props=[split_config.context_prop, split_config.test_prop]
        )

        cls.LOGGER.info("Subsampling training set...")
        train_data = stratified_split(
            train_data,
            default_train_prop=1.0,
            train_props=split_config.subsample_train,
            seed=split_config.data_split_seed,
        ).train

        if split_config.subsample_context:
            cls.LOGGER.info("Subsampling context set...")
            context_data = stratified_split(
                train_data,
                default_train_prop=1.0,
                train_props=split_config.subsample_context,
                seed=split_config.data_split_seed,
            ).train

        # Enable transductive learning (i.e. using the test data for semi-supervised learning)
        if split_config.transductive:
            context_data = context_data.cat(test_data, inplace=False)

        # Assign transforms if datasets are vision ones
        if isinstance(train_data, CdtVisionDataset):
            train_data.transform = split_config.train_transforms
        if isinstance(context_data, CdtVisionDataset):
            context_data.transform = split_config.context_transforms
        if isinstance(test_data, CdtVisionDataset):
            test_data.transform = split_config.test_transforms

        return TrainContextTestSplit(train=train_data, context=context_data, test=test_data)

    @classmethod
    def from_config(cls: Type[Self], config: BaseConfig) -> Self:
        ds_config = config.dm
        split_config = config.split
        dm_config = config.dm

        root = cls.find_data_dir() if ds_config.root is None else ds_config.root
        all_data: D = instantiate(ds_config, root=root, split=None)
        # Use a fraction, governed by ``args.data_pcnt``, of the full dataset
        if split_config.data_prop is not None:
            all_data = stratified_split(all_data, default_train_prop=split_config.data_prop).train
        splits = cls._generate_splits(dataset=all_data, split_config=split_config)

        return cls(train=splits.train, context=splits.context, test=splits.test, **dm_config)

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.context, self.test)

    def __str__(self) -> str:
        ds_name = self.train.__class__.__name__
        size_info = (
            f"Size of training-set: {self.num_train_samples}\n"
            f"Size of context-set: {self.num_context_samples}\n"
            f"size of test-set: {self.num_test_samples}"
        )
        return f"DataModule for dataset of type {ds_name}\n{size_info}"

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
