from typing import NamedTuple

import numpy as np
from torch.utils.data import Dataset, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from ethicml.data import create_genfaces_dataset, create_celeba_dataset
from ethicml.vision.data import LdColorizer

from fdm.configs import BaseArgs

from .adult import load_adult_data
from .dataset_wrappers import LdAugmentedDataset
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
        base_aug = [transforms.ToTensor()]
        data_aug = []
        if args.rotate_data:
            data_aug.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            data_aug.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))
        if args.padding > 0:
            base_aug.insert(0, transforms.Pad(args.padding))
        if args.quant_level != "8":
            base_aug.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            base_aug.append(NoisyDequantize(int(args.quant_level)))
        train_data = MNIST(root=args.root, download=True, train=True)
        test_data = MNIST(root=args.root, download=True, train=False)

        num_classes = 10
        if args.filter_labels:
            num_classes = len(args.filter_labels)
            num_classes = 1 if num_classes == 2 else num_classes

            def _filter(dataset: MNIST):
                targets: np.ndarray[np.int64] = dataset.targets.numpy()
                final_mask = np.zeros_like(targets, dtype=np.bool_)
                for index, label in enumerate(args.filter_labels):
                    mask = targets == label
                    targets = np.where(mask, index, targets)
                    final_mask |= mask
                dataset.targets = targets
                return Subset(dataset, final_mask.nonzero()[0])

            train_data, test_data = _filter(train_data), _filter(test_data)

        context_len = round(args.context_pcnt * len(train_data))
        train_len = len(train_data) - context_len
        context_data, train_data = random_split(train_data, lengths=(context_len, train_len))

        colorizer = LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
        )

        context_data = LdAugmentedDataset(
            context_data,
            ld_augmentations=colorizer,
            num_classes=num_classes,
            li_augmentation=True,
            base_augmentations=data_aug + base_aug,
        )
        train_data = LdAugmentedDataset(
            train_data,
            ld_augmentations=colorizer,
            num_classes=num_classes,
            li_augmentation=False,
            base_augmentations=data_aug + base_aug,
        )
        test_data = LdAugmentedDataset(
            test_data,
            ld_augmentations=colorizer,
            num_classes=num_classes,
            li_augmentation=True,
            base_augmentations=base_aug,
        )

        args._y_dim = num_classes
        args._s_dim = num_classes

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
