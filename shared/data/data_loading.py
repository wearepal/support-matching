import platform
from typing import Dict, Literal, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, Subset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ethicml.data import create_celeba_dataset, create_genfaces_dataset, celeba
from ethicml.vision import TorchImageDataset
from ethicml.vision.data import LdColorizer
from shared.configs import BaseArgs

from .adult import load_adult_data
from .misc import shrink_dataset
from .transforms import NoisyDequantize, Quantize

__all__ = ["DatasetTriplet", "load_dataset"]


class DatasetTriplet(NamedTuple):
    context: Dataset
    test: Dataset
    train: Dataset
    s_dim: int
    y_dim: int


class RawDataTuple(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


def load_dataset(args: BaseArgs) -> DatasetTriplet:
    context_data: Dataset
    test_data: Dataset
    train_data: Dataset
    data_root = args.root or find_data_dir()

    # =============== get whole dataset ===================
    if args.dataset == "cmnist":
        augs = []
        if args.padding > 0:
            augs.append(nn.ConstantPad2d(padding=args.padding, value=0))
        if args.quant_level != "8":
            augs.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            augs.append(NoisyDequantize(int(args.quant_level)))

        train_data = MNIST(root=data_root, download=True, train=True)
        test_data = MNIST(root=data_root, download=True, train=False)

        num_classes = 10
        if args.filter_labels:
            num_classes = len(args.filter_labels)

            def _filter_(dataset: MNIST):
                final_mask = torch.zeros_like(dataset.targets).bool()
                for index, label in enumerate(args.filter_labels):
                    mask = dataset.targets == label
                    dataset.targets[mask] = index
                    final_mask |= mask
                dataset.data = dataset.data[final_mask]
                dataset.targets = dataset.targets[final_mask]

            _filter_(train_data)
            _filter_(test_data)

        num_colors = len(args.colors) if len(args.colors) > 0 else num_classes
        colorizer = LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
            color_indices=args.colors or None,
        )

        test_data = (test_data.data, test_data.targets)
        context_len = round(args.context_pcnt * len(train_data))
        train_len = len(train_data) - context_len
        split_sizes = (context_len, train_len)
        shuffle_inds = torch.randperm(len(train_data))
        context_data, train_data = tuple(
            zip(
                *(
                    train_data.data[shuffle_inds].split(split_sizes),
                    train_data.targets[shuffle_inds].split(split_sizes),
                )
            )
        )

        def _colorize_subset(
            _subset: Tuple[Tensor, Tensor],
            _correlation: float,
            _decorr_op: Literal["random", "shift"],
        ) -> RawDataTuple:
            x, y = _subset
            x = x.unsqueeze(1).expand(-1, 3, -1, -1) / 255.0
            for aug in augs:
                x = aug(x)
            s = y.clone()
            if _decorr_op == "random":  # this is for context and test set
                indexes = torch.rand(s.shape) > _correlation
                s[indexes] = torch.randint_like(s[indexes], low=0, high=num_colors)
            elif args.missing_s:  # this is one possibility for training set
                s = torch.randint_like(s, low=0, high=num_colors)
                for to_remove in args.missing_s:
                    s[s == to_remove] = (to_remove + 1) % num_colors
            else:  # this is another possibility for training set
                indexes = torch.rand(s.shape) > _correlation
                s[indexes] = torch.fmod(s[indexes] + 1, num_colors)
            x_col = colorizer(x, s)
            return RawDataTuple(x=x_col, s=s, y=y)

        def _subsample_by_s_and_y(
            _data: RawDataTuple, _target_props: Dict[int, float]
        ) -> RawDataTuple:
            _x = _data.x
            _s = _data.s
            _y = _data.y
            for _class_id, _prop in _target_props.items():
                assert 0 <= _prop <= 1, "proportions should be between 0 and 1"
                target_y = _class_id // num_classes
                target_s = _class_id % num_colors
                _indexes = (_y == int(target_y)) & (_s == int(target_s))
                _n_matches = len(_indexes.nonzero())
                _to_keep = torch.randperm(_n_matches) < (round(_prop * (_n_matches - 1)))
                _indexes[_indexes.nonzero()[_to_keep]] = False
                _x = _x[~_indexes]
                _s = _s[~_indexes]
                _y = _y[~_indexes]
            return RawDataTuple(x=_x, s=_s, y=_y)

        if args.subsample_train:
            if args.missing_s:
                raise RuntimeError("Don't use subsample_train and missing_s together!")
            # when we manually subsample the training set, we ignore color correlation
            train_data_t = _colorize_subset(train_data, _correlation=0, _decorr_op="random",)
            train_data_t = _subsample_by_s_and_y(train_data_t, args.subsample_train)
        else:
            train_data_t = _colorize_subset(
                train_data, _correlation=args.color_correlation, _decorr_op="shift",
            )
        test_data_t = _colorize_subset(test_data, _correlation=0, _decorr_op="random")
        context_data_t = _colorize_subset(context_data, _correlation=0, _decorr_op="random")

        if args.subsample_context:
            context_data_t = _subsample_by_s_and_y(context_data_t, args.subsample_context)
            # test data remains balanced
            # test_data = _subsample_by_class(*test_data, args.subsample)

        train_data = TensorDataset(train_data_t.x, train_data_t.s, train_data_t.y)
        test_data = TensorDataset(test_data_t.x, test_data_t.s, test_data_t.y)
        context_data = TensorDataset(context_data_t.x, context_data_t.s, context_data_t.y)

        args._y_dim = 1 if num_classes == 2 else num_classes
        args._s_dim = 1 if num_colors == 2 else num_colors

    elif args.dataset == "celeba":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))

        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        # unbiased_pcnt = args.test_pcnt + args.context_pcnt
        dataset, base_dir = celeba(
            download_dir=data_root,
            label=args.celeba_target_attr,
            sens_attr=args.celeba_sens_attr,
            download=True,
            check_integrity=True,
        )
        assert dataset is not None
        all_data = TorchImageDataset(
            data=dataset.load(),
            root=base_dir,
            map_to_binary=True,
            transform=transform,
            target_transform=None,
        )

        size = len(all_data)
        context_len = round(args.context_pcnt * size)
        test_len = round(args.test_pcnt * size)
        train_len = size - context_len - test_len

        context_inds, train_inds, test_inds = torch.randperm(size).split(
            (context_len, train_len, test_len)
        )

        args._y_dim = 1
        args._s_dim = all_data.s_dim

        def _subsample_inds_by_s_and_y(
            _data: TorchImageDataset, _subset_inds: Tensor, _target_props: Dict[int, float]
        ) -> Tensor:
            _s = _data.sens_attr
            _y = _data.target_attr
            _y_dim = max(2, args._y_dim)
            _s_dim = max(2, args._s_dim)

            for _class_id, _prop in _target_props.items():
                assert 0 <= _prop <= 1, "proportions should be between 0 and 1"
                target_y = _class_id // _y_dim
                target_s = _class_id % _s_dim
                import pdb; pdb.set_trace()
                _indexes = (_y == int(target_y)) & (_s == int(target_s)) & _subset_inds
                _to_drop = _indexes & (np.random.uniform(len(_indexes)) < (1 - _prop))
                _subset_inds[_to_drop] = False

            return _subset_inds

        if args.subsample_context:
            context_inds = _subsample_inds_by_s_and_y(
                all_data, context_inds, args.subsample_context
            )
        if args.subsample_train:
            train_inds = _subsample_inds_by_s_and_y(all_data, train_inds, args.subsample_train)

        context_data = Subset(all_data, context_inds)
        train_data = Subset(all_data, train_inds)
        test_data = Subset(all_data, test_inds)

    elif args.dataset == "genfaces":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        unbiased_pcnt = args.test_pcnt + args.context_pcnt
        unbiased_data = create_genfaces_dataset(
            root=data_root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=False,
            mixing_factor=args.mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        context_len = round(args.context_pcnt / unbiased_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - context_len
        context_data, test_data = random_split(unbiased_data, lengths=(context_len, test_len))

        train_data = create_genfaces_dataset(
            root=data_root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=True,
            mixing_factor=args.mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args._y_dim = 1
        args._s_dim = unbiased_data.s_dim

    elif args.dataset == "adult":
        context_data, train_data, test_data = load_adult_data(args)
        args._y_dim = 1
        if args.adult_split == "Education":
            args._s_dim = 3
        elif args.adult_split == "Sex":
            args._s_dim = 1
        else:
            raise ValueError(f"This split is not yet fully supported: {args.adult_split}")
    else:
        raise ValueError("Invalid choice of dataset.")

    if 0 < args.data_pcnt < 1:
        context_data = shrink_dataset(context_data, args.data_pcnt)
        train_data = shrink_dataset(train_data, args.data_pcnt)
        test_data = shrink_dataset(test_data, args.data_pcnt)

    return DatasetTriplet(
        context=context_data,
        test=test_data,
        train=train_data,
        s_dim=args._s_dim,
        y_dim=args._y_dim,
    )


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    data_dirs = {
        "fear": "/mnt/data0/data",
        "hydra": "/mnt/archive/shared/data",
        "m900382.inf.susx.ac.uk": "/Users/tk324/PycharmProjects/NoSINN/data",
        "turing": "/srv/galene0/shared/data",
    }
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return data_dirs.get(name_of_machine, "data")
