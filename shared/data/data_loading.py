from pathlib import Path
from typing import Literal, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Subset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ethicml.data import create_celeba_dataset, create_genfaces_dataset
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


def load_dataset(args: BaseArgs) -> DatasetTriplet:
    context_data: Dataset
    test_data: Dataset
    train_data: Dataset

    # =============== get whole dataset ===================
    if args.dataset == "cmnist":
        augs = []
        if args.padding > 0:
            augs.append(lambda x: F.pad(x, (args.padding, args.padding)))
        if args.quant_level != "8":
            augs.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            augs.append(NoisyDequantize(int(args.quant_level)))

        train_data = MNIST(root=args.root, download=True, train=True)
        test_data = MNIST(root=args.root, download=True, train=False)

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

        colorizer = LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
            color_indices=args.filter_labels or None,
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

        if args.subsample:

            def _subsample_by_class(
                _data: Tensor, _targets: Tensor, _target_props: Dict[int, float]
            ) -> Tuple[Tensor, Tensor]:
                for _target, _prop in _target_props.items():
                    assert 0 <= _prop <= 1
                    _indexes = _targets == int(_target)
                    _n_matches = len(_indexes.nonzero())
                    _to_keep = torch.randperm(_n_matches) < (round(_prop * (_n_matches - 1)))
                    _indexes[_indexes.nonzero()[_to_keep]] = False
                    _data = _data[~_indexes]
                    _targets = _targets[~_targets]
                return _data, _targets

            context_data = _subsample_by_class(*context_data, args.subsample)
            test_data = _subsample_by_class(*test_data, args.subsample)

        def _colorize_subset(
            _subset: Tuple[Tensor, Tensor],
            _correlation: float,
            _decorr_op: Literal["random", "shift"],
        ) -> TensorDataset:
            x, y = _subset
            x = x.unsqueeze(1).expand(-1, 3, -1, -1) / 255.0
            for aug in augs:
                x = aug(x)
            s = y.clone()
            indexes = torch.rand(s.shape) > _correlation
            if _decorr_op == "random":
                s[indexes] = torch.randint_like(s[indexes], low=0, high=num_classes)
            else:
                s[indexes] = torch.fmod(s[indexes] + 1, num_classes)
            x_col = colorizer(x, s)
            return TensorDataset(x_col, s, y)

        train_data = _colorize_subset(
            train_data, _correlation=args.color_correlation, _decorr_op="shift",
        )
        test_data = _colorize_subset(test_data, _correlation=0, _decorr_op="random")
        context_data = _colorize_subset(context_data, _correlation=0, _decorr_op="random")

        args._y_dim = 1 if num_classes == 2 else num_classes
        args._s_dim = 1 if num_classes == 2 else num_classes

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

        unbiased_pcnt = args.test_pcnt + args.context_pcnt
        unbiased_data = create_celeba_dataset(
            root=args.root,
            sens_attr_name=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
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

        train_data = create_celeba_dataset(
            root=args.root,
            sens_attr_name=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
            biased=True,
            mixing_factor=args.mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args._y_dim = 1
        args._s_dim = unbiased_data.s_dim

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
            root=args.root,
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
            root=args.root,
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
        args._s_dim = 1
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
