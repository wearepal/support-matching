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
from shared.utils.utils import lcm

from .utils import group_id_to_label

__all__ = [
    "D",
    "DataModule",
    "Dataset",
    "TrainDepSplit",
]


Dataset: TypeAlias = CdtDataset[TernarySample[LoadedData], Any, Tensor, Tensor]
D = TypeVar("D", bound=Dataset)


@attr.define(kw_only=True)
class TrainDepSplit(Generic[D]):
    train: D
    deployment: D
    test: D

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.deployment, self.test)

    def num_samples(self) -> int:
        return len(self.train) + len(self.deployment) + len(self.test)

    @property
    def num_samples_tr(self) -> int:
        return len(self.train)

    @property
    def num_samples_dep(self) -> int:
        return len(self.deployment)

    @property
    def num_samples_te(self) -> int:
        return len(self.test)


@attr.define(kw_only=True)
class DataModule(Generic[D]):
    LOGGER: ClassVar[logging.Logger] = init_logger("DataModule")

    DATA_DIRS: ClassVar[Dict[str, str]] = {
        "turing": "/srv/galene0/shared/data",
        "fear": "/srv/galene0/shared/data",
        "hydra": "/srv/galene0/shared/data",
        "goedel": "/srv/galene0/shared/data",
    }

    train: D
    deployment: D
    test: D

    # DataLoader settings
    batch_size_tr: int
    _batch_size_te: Optional[int] = None
    num_samples_per_group_per_bag: int = 1

    num_workers: int = 4
    persist_workers: bool = False
    pin_memory: bool = True
    seed: int = 47

    gt_deployment: bool = False
    label_noise: float = attr.field(default=0)
    generator: torch.Generator = attr.field(init=False)

    @label_noise.validator  # type: ignore
    def validate_label_noise(self, attribute: str, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"'{attribute}' must be in the range [0, 1].")

    def __attrs_post_init__(self) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)

    @property
    def batch_size_te(self) -> int:
        return self.batch_size_tr if self._batch_size_te is None else self._batch_size_te

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
    def num_dep_samples(self) -> int:
        return len(self.deployment)

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
        num_workers: Optional[int] = None,
    ) -> CdtDataLoader[TernarySample]:
        """Make DataLoader."""
        return CdtDataLoader(
            ds,
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers if num_workers is None else num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            generator=self.generator,
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
            generator=self.generator,
        )

    def train_dataloader(
        self,
        eval: bool = False,
        *,
        balance: bool = True,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
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
            num_workers=num_workers,
        )

    @staticmethod
    def _inject_label_noise(
        labels: Tensor,
        *,
        noise_level: float,
        generator: torch.Generator,
        inplace: bool = True,
    ) -> Tensor:
        if not 0 <= noise_level <= 1:
            raise ValueError("Noise-level must be in the range [0, 1].")
        if not inplace:
            labels = labels.clone()
        unique, unique_inv = labels.unique(return_inverse=True)
        num_to_flip = round(noise_level * len(labels))
        to_flip = torch.randperm(len(labels), generator=generator)[:num_to_flip]
        unique_inv[to_flip] += torch.randint(low=1, high=len(unique), size=(num_to_flip,))
        unique_inv[to_flip] %= len(unique)
        return unique[unique_inv]

    def deployment_dataloader(
        self,
        group_ids: Optional[Tensor] = None,
        *,
        eval: bool = False,
        num_workers: Optional[int] = None,
    ) -> CdtDataLoader[TernarySample]:
        if eval:
            return self._make_dataloader(
                ds=self.deployment, batch_size=self.batch_size_te, shuffle=False
            )

        # Use the ground-truth y/s labels for stratified sampling
        if self.gt_deployment:
            group_ids = get_group_ids(self.deployment)
            # Inject label-noise into the group identifiers.
            if self.label_noise > 0:
                group_ids = self._inject_label_noise(
                    group_ids, noise_level=self.label_noise, generator=self.generator
                )

        if group_ids is None:
            batch_sampler = SequentialBatchSampler(
                data_source=self.deployment,
                batch_size=self.batch_size_tr,
                shuffle=True,
                training_mode=TrainingMode.step,
                drop_last=False,
                generator=self.generator,
            )
        else:
            batch_sampler = self._make_stratified_sampler(
                group_ids=group_ids,
                batch_size=self.batch_size_tr,
            )
        return self._make_dataloader(
            ds=self.deployment,
            batch_size=1,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers: Optional[int] = None) -> CdtDataLoader[TernarySample]:
        return self._make_dataloader(
            ds=self.test,
            batch_size=self.batch_size_te,
            shuffle=False,
            num_workers=num_workers,
        )

    @classmethod
    def _generate_splits(cls: Type[Self], dataset: D, split_config: SplitConf) -> TrainDepSplit[D]:

        dep_data, test_data, train_data = dataset.random_split(
            props=[split_config.dep_prop, split_config.test_prop],
            seed=split_config.seed,
        )

        cls.LOGGER.info("Subsampling training set...")
        train_data = stratified_split(
            train_data,
            default_train_prop=1.0,
            train_props=split_config.train_subsampling_props,
            seed=split_config.seed,
        ).train

        if split_config.dep_subsampling_props:
            cls.LOGGER.info("Subsampling deployment set...")
            dep_data = stratified_split(
                dep_data,
                default_train_prop=1.0,
                train_props=split_config.dep_subsampling_props,
                seed=split_config.seed,
            ).train

        # Enable transductive learning (i.e. using the test data for semi-supervised learning)
        if split_config.transductive:
            dep_data = dep_data.cat(test_data, inplace=False)

        # Assign transforms if datasets are vision ones
        if isinstance(train_data, CdtVisionDataset):
            train_data.transform = (
                cls._default_train_transforms()
                if split_config.train_transforms is None
                else split_config.train_transforms
            )
        if isinstance(dep_data, CdtVisionDataset):
            dep_data.transform = split_config.dep_transforms
            train_data = cast(CdtVisionDataset, train_data)
            dep_data.transform = (
                train_data.transform
                if split_config.dep_transforms is None
                else split_config.dep_transforms
            )
        if isinstance(test_data, CdtVisionDataset):
            test_data.transform = split_config.test_transforms
            test_data.transform = (
                cls._default_test_transforms()
                if split_config.test_transforms is None
                else split_config.test_transforms
            )

        return TrainDepSplit(train=train_data, deployment=dep_data, test=test_data)

    @classmethod
    def from_configs(
        cls: Type[Self],
        *,
        dm_config: DataModuleConf,
        ds_config: DictConfig,
        split_config: SplitConf,
    ) -> Self:
        split_config = instantiate(split_config)

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
            deployment=splits.deployment,
            test=splits.test,
            **dm_config,  # type: ignore
        )

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.deployment, self.test)

    def __str__(self) -> str:
        ds_name = self.train.__class__.__name__
        size_info = (
            f"- Size of training-set: {self.num_train_samples}\n"
            f"- Size of deployment-set: {self.num_dep_samples}\n"
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
                        inner_tforms = _tform.transforms
                        for inner_tform in inner_tforms[::-1]:
                            stats = _get_stats(inner_tform)
                            if stats is not None:
                                break
                    return stats

                if (stats := _get_stats(tform)) is not None:
                    return denormalize(x, mean=stats.mean, std=stats.std, inplace=inplace)
        return x
