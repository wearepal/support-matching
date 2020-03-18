"""Definition of the Adult dataset"""
from typing import Tuple

import numpy as np

from ethicml.data import Adult, load_data

from fdm.configs import SharedArgs
from .adult import Triplet, biased_split
from .dataset_wrappers import DataTupleDataset, PerturbedDataTupleDataset

__all__ = ["load_perturbed_adult"]


def load_perturbed_adult(
    args: SharedArgs,
) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    adult_dataset = Adult(binarize_nationality=args.drop_native)
    data = load_data(adult_dataset, ordered=True, generate_dummies=True)

    cont_feats = adult_dataset.continuous_features

    # construct a new x in which all discrete features are "continuous"
    new_x = data.x[cont_feats]
    all_feats = cont_feats
    disc_feature_groups = adult_dataset.disc_feature_groups
    assert disc_feature_groups is not None
    for name, group in disc_feature_groups.items():
        one_hot = data.x[group].to_numpy(np.int64)
        indexes = np.argmax(one_hot, axis=1)
        new_x = new_x.assign(**{name: indexes})
        all_feats.append(name)

    # number of bins
    num_bins = new_x.max().to_numpy() + 1
    # normalize
    new_x = new_x / num_bins

    data = data.replace(x=new_x)

    tuples: Triplet = biased_split(args, data)
    meta_train, task, task_train = tuples.meta, tuples.task, tuples.task_train

    pretrain_data = PerturbedDataTupleDataset(meta_train, features=all_feats, num_bins=num_bins)
    train_data = PerturbedDataTupleDataset(task_train, features=all_feats, num_bins=num_bins)
    test_data = PerturbedDataTupleDataset(task, features=all_feats, num_bins=num_bins)
    return pretrain_data, train_data, test_data
