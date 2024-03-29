from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial, reduce
from math import gcd
from typing import TYPE_CHECKING
from typing_extensions import Self

import albumentations as A
from conduit.data import IMAGENET_STATS
from conduit.data.datasets import CdtDataLoader, get_group_ids
from conduit.data.datasets.vision import CdtVisionDataset, ImageTform, PillowTform
from conduit.data.structures import MeanStd, TernarySample
from conduit.transforms.image import denormalize
from loguru import logger
import pandas as pd
from ranzen import gcopy
from ranzen.torch.data import (
    ApproxStratBatchSampler,
    BaseSampler,
    BatchSamplerBase,
    SequentialBatchSampler,
    StratifiedBatchSampler,
    TrainingMode,
)
import torch
from torch import Tensor
import torchvision.transforms.transforms as T

from .common import Dataset
from .splitter import DataSplitter
from .utils import group_id_to_label

if TYPE_CHECKING:
    from src.labelling import Labeller

__all__ = ["DataModule", "DataModuleConf"]


def lcm(denominators: Iterable[int]) -> int:
    """Least common multiplier."""
    return reduce(lambda a, b: a * b // gcd(a, b), denominators)


class StratSamplerType(Enum):
    """How is stratified batch sampling realized?"""

    exact = auto()
    """Each bag contains all groups."""
    approx_group = auto()
    """For each class, sample as many subgroups as there are subgroups in total."""
    approx_class = auto()
    """Only sample one subgroup per class."""


@dataclass
class DataModuleConf:
    """DataModule settings that are configurable by hydra."""

    batch_size_tr: int = 1
    batch_size_te: int | None = None
    num_samples_per_group_per_bag: int = 1
    stratified_sampler: StratSamplerType = StratSamplerType.exact
    use_y_for_dep_bags: bool = False
    """If True, the code may use ground-truth y labels to construct stratified deployment bags."""

    # DataLoader settings
    num_workers: int = 0
    persist_workers: bool = False
    pin_memory: bool = True
    seed: int = 47


@dataclass(eq=False)
class DataModule:
    cfg: DataModuleConf
    train: Dataset
    deployment: Dataset
    deployment_ids: Tensor | None = field(init=False, default=None)
    test: Dataset
    split_seed: int | None

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
    def batch_size_te(self, value: int | None) -> None:
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

    def set_deployment_labels(self, ids: Tensor | None = None) -> Self:
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
    def missing_sources(self) -> set[int]:
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
    def feature_group_slices(self) -> dict[str, list[slice]] | None:
        return None

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

    def _make_dataloader(
        self,
        ds: Dataset,
        *,
        batch_size: int | None,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: BatchSamplerBase | None = None,
        num_workers: int | None = None,
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
    def get_group_multipliers(group_ids: Tensor, *, card_s: int) -> dict[int, int]:
        """This is a standalone function only because then we can have a unit test for it."""
        unique_ids = group_ids.unique(sorted=False).tolist()

        # first, count how many subgroups there are for each y
        num_subgroups_per_y: defaultdict[int, int] = defaultdict(int)
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

    def _get_balanced_sampler(
        self, group_ids: Tensor, *, batch_size: int
    ) -> StratifiedBatchSampler | ApproxStratBatchSampler:
        if self.cfg.stratified_sampler is StratSamplerType.exact:
            return self._make_stratified_sampler(group_ids, batch_size=batch_size)

        num_samples_effective = self.cfg.num_samples_per_group_per_bag * batch_size
        if self.cfg.stratified_sampler is StratSamplerType.approx_group:
            batch_sampler_fn = partial(
                ApproxStratBatchSampler, num_samples_per_group=num_samples_effective
            )
        else:
            batch_sampler_fn = partial(
                ApproxStratBatchSampler, num_samples_per_class=num_samples_effective
            )
        # It's a bit hacky that we're re-computing the s and y labels from the group IDs,
        # but it has to be done this way for the label noiser to work.
        labels = group_id_to_label(group_ids, s_count=self.card_s)
        return batch_sampler_fn(
            class_labels=labels.y.flatten().tolist(),
            subgroup_labels=labels.s.flatten().tolist(),
            training_mode=TrainingMode.step,
            generator=self.generator,
        )

    def _make_stratified_sampler(
        self, group_ids: Tensor, *, batch_size: int
    ) -> StratifiedBatchSampler:
        multipliers = self.get_group_multipliers(group_ids, card_s=self.test.card_s)
        num_samples_per_group = self.cfg.num_samples_per_group_per_bag * batch_size
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
        batch_size: int | None = None,
        num_workers: int | None = None,
        batch_sampler: BatchSamplerBase | None = None,
    ) -> CdtDataLoader[TernarySample]:
        if eval:
            return self._make_dataloader(
                ds=self.train, batch_size=self.batch_size_te, shuffle=False, num_workers=num_workers
            )
        batch_size = self.batch_size_tr if batch_size is None else batch_size
        if batch_sampler is None:
            if balance:
                batch_sampler = self._get_balanced_sampler(self.group_ids_tr, batch_size=batch_size)
            else:
                batch_sampler = SequentialBatchSampler(
                    data_source=self.train,
                    batch_size=batch_size,
                    shuffle=True,
                    training_mode=TrainingMode.step,
                    drop_last=False,
                    generator=self.generator,
                )
            logger.info(f"effective batch size: {batch_sampler.batch_size}")
        return self._make_dataloader(
            ds=self.train, batch_size=1, batch_sampler=batch_sampler, num_workers=num_workers
        )

    def deployment_dataloader(
        self,
        *,
        eval: bool = False,
        num_workers: int | None = None,
        batch_size: int | None = None,
    ) -> CdtDataLoader[TernarySample]:
        batch_size = self.batch_size_tr if batch_size is None else batch_size
        if eval:
            return self._make_dataloader(ds=self.deployment, batch_size=batch_size, shuffle=False)

        batch_sampler: SequentialBatchSampler | StratifiedBatchSampler | ApproxStratBatchSampler
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
            if self.cfg.use_y_for_dep_bags:
                batch_sampler = self._get_balanced_sampler(
                    self.deployment_ids, batch_size=batch_size
                )
            else:
                if self.cfg.stratified_sampler is not StratSamplerType.exact:
                    logger.info(
                        "warning: train batches and deployment batches"
                        " are using different batch samplers"
                    )
                batch_sampler = self._make_stratified_sampler(
                    self.deployment_ids, batch_size=batch_size
                )
        logger.info(f"effective batch size: {batch_sampler.batch_size}")
        return self._make_dataloader(
            ds=self.deployment, batch_size=1, batch_sampler=batch_sampler, num_workers=num_workers
        )

    def test_dataloader(self, num_workers: int | None = None) -> CdtDataLoader[TernarySample]:
        return self._make_dataloader(
            ds=self.test, batch_size=self.batch_size_te, shuffle=False, num_workers=num_workers
        )

    @property
    def transforms_tr(self) -> ImageTform | None:
        if isinstance(self.train, CdtVisionDataset):
            return self.train.transform
        return None

    @transforms_tr.setter
    def transforms_tr(self, value: ImageTform | None) -> None:
        if isinstance(self.train, CdtVisionDataset):
            self.train.transform = self._default_train_transforms() if value is None else value

    @property
    def transforms_dep(self) -> ImageTform | None:
        if isinstance(self.deployment, CdtVisionDataset):
            return self.deployment.transform
        return None

    @transforms_dep.setter
    def transforms_dep(self, value: ImageTform | None) -> None:
        if isinstance(self.deployment, CdtVisionDataset):
            assert isinstance(self.train, CdtVisionDataset)
            self.deployment.transform = self.train.transform if value is None else value

    @property
    def transforms_te(self) -> ImageTform | None:
        if isinstance(self.test, CdtVisionDataset):
            return self.test.transform
        return None

    @transforms_te.setter
    def transforms_te(self, value: ImageTform | None) -> None:
        if isinstance(self.test, CdtVisionDataset):
            self.test.transform = self._default_test_transforms() if value is None else value

    def set_transforms_all(self, value: ImageTform | None) -> None:
        self.transforms_tr = value
        self.transforms_te = value
        self.transforms_dep = value

    @classmethod
    def from_ds(
        cls,
        *,
        config: DataModuleConf,
        ds: Dataset,
        splitter: DataSplitter,
        labeller: "Labeller",
        device: torch.device,
    ) -> Self:
        splits = splitter(ds)
        dm = cls(
            cfg=config,
            train=splits.train,
            deployment=splits.deployment,
            test=splits.test,
            split_seed=getattr(splitter, "seed", None),
        )
        deployment_ids = labeller.run(dm=dm, device=device)
        dm.deployment_ids = deployment_ids
        return dm

    def __iter__(self) -> Iterator[Dataset]:
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

                def _get_stats(_tform: ImageTform) -> MeanStd | None:
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

    def print_statistics(self) -> None:
        y_names = {0: "No", 1: "Yes"}
        s_names = {0: "Female", 1: "Male"}
        dfs: dict[str, pd.DataFrame] = {}
        for data, name in [
            (self.train, "Train"),
            (self.deployment, "Deployment"),
            # (self.test, "Test"),
        ]:
            ys = data.y
            ss = data.s
            total = len(ys)

            df = pd.DataFrame(
                {
                    "Smiling": pd.Series(dtype="str"),
                    "Gender": pd.Series(dtype="str"),
                    "Number": pd.Series(dtype="int"),
                    "Fraction": pd.Series(dtype="str"),
                }
            )
            i = 0
            for y in ys.unique():
                for s in ss.unique():
                    num = int(torch.sum((ys == y) * (ss == s)))
                    df.loc[i] = (  # type: ignore
                        y_names[int(y)],
                        s_names[int(s)],
                        num,
                        f"{(100 * num / total):.3g}\\%",
                    )
                    i += 1
            dfs[name] = df
        table = pd.concat(dfs, axis=1)
        print(
            table.to_latex(
                index=False,
                # float_format="%.5g",
                multicolumn_format="l",
            )
        )
