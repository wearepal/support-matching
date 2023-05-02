from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from typing import (
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self, override

import albumentations as A
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import CdtDataLoader, get_group_ids
from conduit.data.datasets.vision import CdtVisionDataset, ImageTform, PillowTform
from conduit.data.structures import MeanStd, TernarySample
from conduit.transforms.image import denormalize
from loguru import logger
from ranzen import gcopy
from ranzen.torch.data import (
    BaseSampler,
    BatchSamplerBase,
    SequentialBatchSampler,
    StratifiedBatchSampler,
    TrainingMode,
    num_batches_per_epoch,
)
import torch
from torch import Tensor
import torchvision.transforms.transforms as T

from .common import D
from .splitter import DataSplitter
from .utils import group_id_to_label

if TYPE_CHECKING:
    from src.labelling import Labeller

__all__ = ["DataModule", "DataModuleConf"]


def lcm(denominators: Iterable[int]) -> int:
    """Least common multiplier."""
    return reduce(lambda a, b: a * b // gcd(a, b), denominators)


@dataclass
class DataModuleConf:
    """DataModule settings that are configurable by hydra."""

    batch_size_tr: int = 1
    batch_size_te: Optional[int] = None
    num_samples_per_group_per_bag: int = 1

    # DataLoader settings
    num_workers: int = 0
    persist_workers: bool = False
    pin_memory: bool = True
    seed: int = 47


@dataclass(eq=False)
class DataModule(Generic[D]):
    cfg: DataModuleConf
    train: D
    deployment: D
    deployment_ids: Optional[Tensor] = field(init=False, default=None)
    test: D
    split_seed: Optional[int]

    def __post_init__(self) -> None:
        # we have to store `batch_size_tr` in `self` because `gcopy` may want to overwrite it
        self.batch_size_tr: int = self.cfg.batch_size_tr
        self.batch_size_te = self.cfg.batch_size_te

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.cfg.seed)

    @property
    def batch_size_te(self) -> int:
        return self.batch_size_tr if self._batch_size_te is None else self._batch_size_te

    @batch_size_te.setter
    def batch_size_te(self, value: Optional[int]) -> None:
        self._batch_size_te = value

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

    def set_deployment_labels(self, ids: Optional[Tensor] = None) -> Self:
        ids = self.deployment_ids if ids is None else ids
        if ids is not None:
            if len(ids) != len(self.deployment):
                raise ValueError(
                    "'ids' must be the same length as the deployment set whose labels are to be "
                    "set."
                )
            s_count = self.card_s
            labels = group_id_to_label(group_id=ids, s_count=s_count)
            y_dep = labels.y.flatten()
            s_dep = labels.s.flatten()
            copy = gcopy(self, deep=True)
            copy.deployment.y = y_dep
            copy.deployment.s = s_dep
            return copy
        logger.warning("No deployment ids to be converted into labels and subsequently set.")
        return self

    def merge_deployment_into_train(self) -> Self:
        if self.deployment_ids is None:
            logger.warning(
                "'train' and 'deployment' sets cannot be merged as the latter is"
                " unlabelled ('deployment_ids=None')"
            )
            return self
        copy = self.set_deployment_labels()
        copy.train += copy.deployment
        return copy

    @property
    def card_s(self) -> int:
        return self.train.card_s

    @property
    def num_sources_tr(self) -> int:
        return len(self.group_ids_tr.unique())

    @property
    def num_sources_dep(self) -> int:
        return len(self.group_ids_dep.unique())

    @property
    def num_sources_te(self) -> int:
        return len(self.group_ids_te.unique())

    @property
    def missing_sources(self) -> Set[int]:
        sources_tr = set(self.group_ids_tr.unique().tolist())
        sources_dep = set(self.group_ids_dep.unique().tolist())
        return sources_dep - sources_tr

    @property
    def num_classes(self) -> int:
        return max(2, self.card_y)

    @property
    def bag_size(self) -> int:
        return self.card_y * self.card_s * self.cfg.num_samples_per_group_per_bag

    @property
    def group_ids_tr(self) -> Tensor:
        return get_group_ids(self.train)

    @property
    def group_ids_dep(self) -> Tensor:
        return get_group_ids(self.deployment)

    @property
    def group_ids_te(self) -> Tensor:
        return get_group_ids(self.test)

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
            num_workers=self.cfg.num_workers if num_workers is None else num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.cfg.persist_workers,
            generator=self.generator,
            batch_sampler=batch_sampler,
        )

    @staticmethod
    def get_group_multipliers(group_ids: Tensor, *, card_s: int) -> Dict[int, int]:
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
        multipliers = self.get_group_multipliers(group_ids, card_s=self.test.card_s)
        num_samples_per_group = self.cfg.num_samples_per_group_per_bag * batch_size
        sam = StratifiedBatchSampler(
            group_ids=group_ids.squeeze().tolist(),
            num_samples_per_group=num_samples_per_group,
            shuffle=True,
            base_sampler=BaseSampler.sequential,
            multipliers=multipliers,
            training_mode=TrainingMode.step,
            drop_last=False,
            generator=self.generator,
        )
        logger.info(
            f"{sam.num_groups_effective=}, {len(sam.groupwise_idxs)=}, {sam.num_samples_per_group=}"
        )
        return sam

    def train_dataloader(
        self,
        eval: bool = False,
        *,
        balance: bool = True,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        batch_sampler: Optional[BatchSamplerBase] = None,
    ) -> CdtDataLoader[TernarySample]:
        if eval:
            return self._make_dataloader(
                ds=self.train,
                batch_size=self.batch_size_te,
                shuffle=False,
                num_workers=num_workers,
            )
        batch_size = self.batch_size_tr if batch_size is None else batch_size
        if batch_sampler is None:
            if balance:
                batch_sampler = self._make_stratified_sampler(
                    group_ids=self.group_ids_tr, batch_size=batch_size
                )
            else:
                batch_sampler = SequentialBatchSampler(
                    data_source=self.train,
                    batch_size=batch_size,
                    shuffle=True,
                    training_mode=TrainingMode.step,
                    drop_last=False,
                    generator=self.generator,
                )
        return self._make_dataloader(
            ds=self.train,
            batch_size=1,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )

    def deployment_dataloader(
        self,
        *,
        eval: bool = False,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> CdtDataLoader[TernarySample]:
        batch_size = self.batch_size_tr if batch_size is None else batch_size
        if eval:
            return self._make_dataloader(ds=self.deployment, batch_size=batch_size, shuffle=False)

        if self.deployment_ids is None:
            batch_sampler = SequentialBatchSampler(
                data_source=self.deployment,
                batch_size=batch_size,
                shuffle=True,
                training_mode=TrainingMode.step,
                drop_last=False,
                generator=self.generator,
            )
        else:
            batch_sampler = self._make_stratified_sampler(
                group_ids=self.deployment_ids,
                batch_size=batch_size,
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

    @property
    def transforms_tr(self) -> Optional[ImageTform]:
        if isinstance(self.train, CdtVisionDataset):
            return self.train.transform
        return None

    @transforms_tr.setter
    def transforms_tr(self, value: Optional[ImageTform]) -> None:
        if isinstance(self.train, CdtVisionDataset):
            self.train.transform = self._default_train_transforms() if value is None else value

    @property
    def transforms_dep(self) -> Optional[ImageTform]:
        if isinstance(self.deployment, CdtVisionDataset):
            return self.deployment.transform
        return None

    @transforms_dep.setter
    def transforms_dep(self, value: Optional[ImageTform]) -> None:
        if isinstance(self.deployment, CdtVisionDataset):
            assert isinstance(self.train, CdtVisionDataset)
            self.deployment.transform = self.train.transform if value is None else value

    @property
    def transforms_te(self) -> Optional[ImageTform]:
        if isinstance(self.test, CdtVisionDataset):
            return self.test.transform
        return None

    @transforms_te.setter
    def transforms_te(self, value: Optional[ImageTform]) -> None:
        if isinstance(self.test, CdtVisionDataset):
            self.test.transform = self._default_test_transforms() if value is None else value

    def set_transforms_all(self, value: Optional[ImageTform]) -> None:
        self.transforms_tr = value
        self.transforms_te = value
        self.transforms_dep = value

    @classmethod
    def from_ds(
        cls, *, config: DataModuleConf, ds: D, splitter: DataSplitter, labeller: "Labeller"
    ) -> Self:
        splits = splitter(ds)
        dm = cls(
            cfg=config,
            train=splits.train,
            deployment=splits.deployment,
            test=splits.test,
            split_seed=getattr(splitter, "seed", None),
        )
        deployment_ids = labeller.run(dm=dm)
        dm.deployment_ids = deployment_ids
        return dm

    def __iter__(self) -> Iterator[D]:
        yield from (self.train, self.deployment, self.test)

    def __str__(self) -> str:
        ds_name = self.train.__class__.__name__
        size_info = (
            f"- Size of training-set: {self.num_train_samples}\n"
            f"- Size of deployment-set: {self.num_dep_samples}\n"
            f"- Size of test-set: {self.num_test_samples}\n"
            f"- Missing source(s): {self.missing_sources}"
        )
        return f"\nDataModule for dataset of type '{ds_name}'\n{size_info}"

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


class ApproxStratBatchSampler(BatchSamplerBase):
    """Approximate Stratified Batch Sampler.

    For each class, we just sample n=card_s samples and hope that this covers all subgroups.
    """

    def __init__(
        self,
        class_ids: Sequence[int],
        num_samples_per_group: int,
        card_s: int,
        trainig_mode: TrainingMode = TrainingMode.step,
        drop_last: bool = False,
        generator: Union[torch.Generator, None] = None,
    ) -> None:
        class_ids_t = torch.as_tensor(class_ids, dtype=torch.int64)
        classes: list[int] = class_ids_t.unique().tolist()

        classwise_idxs: list[Tensor] = []
        for class_ in classes:
            idxs = (class_ids_t == class_).nonzero(as_tuple=False).view(-1)
            classwise_idxs.append(idxs)

            if len(idxs) < num_samples_per_group * card_s:
                raise ValueError(
                    f"Not enough samples in class {class_} to sample {num_samples_per_group}"
                )

        self.classwise_idxs = classwise_idxs
        self.card_s = card_s
        self.num_samples_per_group = num_samples_per_group
        self.training_mode = trainig_mode
        self.generator = generator

        if self.training_mode is TrainingMode.epoch:
            # some classes have fewer samples than others
            # we want to know how many batches to sample to cover the biggest class
            classwise_epoch_lengths = [
                num_batches_per_epoch(
                    num_samples=len(idxs),
                    batch_size=card_s * num_samples_per_group,
                    drop_last=drop_last,
                )
                for idxs in self.classwise_idxs
            ]
            max_epoch_length = max(classwise_epoch_lengths)
        else:
            max_epoch_length = None

        super().__init__(epoch_length=max_epoch_length)

    @override
    def __iter__(self) -> Iterator[list[int]]:
        generator = self.generator
        if generator is None:
            generator = torch.Generator()
            generator = generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        while True:
            sampled_idxs: list[Tensor] = []
            for class_idxs in self.classwise_idxs:
                # first shuffle the indexes and then take as many as we need
                shuffled_idx = class_idxs[torch.randperm(len(class_idxs), generator=generator)]
                sampled_idxs.append(shuffled_idx[: self.num_samples_per_group * self.card_s])
            yield torch.cat(sampled_idxs, dim=0).tolist()
