"""Definition of the Adult dataset"""
from typing import NamedTuple, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ethicml.data import Adult, load_data
from ethicml.preprocessing import (
    get_biased_and_debiased_subsets,
    get_biased_subset,
    train_test_split,
)
from ethicml.utility import DataTuple
from ethicml.utility.data_helpers import shuffle_df
from fdm.configs import SharedArgs

from .dataset_wrappers import DataTupleDataset

__all__ = ["get_data_tuples", "load_adult_data", "pytorch_data_to_dataframe"]


class Triplet(NamedTuple):
    """Small helper class; basically for enabling named returns"""

    meta: DataTuple
    task: DataTuple
    task_train: DataTuple


def load_adult_data_tuples(args: SharedArgs) -> Tuple[Triplet, Dict[str, List[str]]]:
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    adult_dataset = Adult(binarize_nationality=args.drop_native)
    data = load_data(adult_dataset, ordered=True, generate_dummies=True)

    disc_feature_groups = adult_dataset.disc_feature_groups
    assert disc_feature_groups is not None
    cont_feats = adult_dataset.continuous_features

    tuples: Triplet = biased_split(args, data)
    meta_train, task, task_train = tuples.meta, tuples.task, tuples.task_train

    scaler = StandardScaler()

    task_train_scaled = task_train.x
    task_train_scaled[cont_feats] = scaler.fit_transform(
        task_train.x[cont_feats].to_numpy(np.float32)
    )
    if args.drop_discrete:
        task_train = task_train.replace(x=task_train_scaled[cont_feats])
        disc_feature_groups = {}
    else:
        task_train = task_train.replace(x=task_train_scaled)

    task_scaled = task.x
    task_scaled[cont_feats] = scaler.transform(task.x[cont_feats].to_numpy(np.float32))
    if args.drop_discrete:
        task = task.replace(x=task_scaled[cont_feats])
    else:
        task = task.replace(x=task_scaled)

    meta_train_scaled = meta_train.x
    meta_train_scaled[cont_feats] = scaler.transform(meta_train.x[cont_feats].to_numpy(np.float32))
    if args.drop_discrete:
        meta_train = meta_train.replace(x=meta_train_scaled[cont_feats])
    else:
        meta_train = meta_train.replace(x=meta_train_scaled)

    return Triplet(meta=meta_train, task=task, task_train=task_train), disc_feature_groups


def load_adult_data(
    args: SharedArgs,
) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    tuples, disc_feature_groups = load_adult_data_tuples(args)
    pretrain_tuple, test_tuple, train_tuple = tuples.meta, tuples.task, tuples.task_train
    assert pretrain_tuple is not None
    source_dataset = Adult()
    cont_features = source_dataset.continuous_features
    pretrain_data = DataTupleDataset(
        pretrain_tuple, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    train_data = DataTupleDataset(
        train_tuple, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    test_data = DataTupleDataset(
        test_tuple, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    return pretrain_data, train_data, test_data


def biased_split(args: SharedArgs, data: DataTuple) -> Triplet:
    """Split the dataset such that the task subset is very biased"""
    use_new_split = True
    if use_new_split:
        task_train_tuple, unbiased = get_biased_subset(
            data=data,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.pretrain_pcnt,
            seed=args.data_split_seed,
            data_efficient=True,
        )
    else:
        task_train_tuple, unbiased = get_biased_and_debiased_subsets(
            data=data,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.pretrain_pcnt,
            seed=args.data_split_seed,
        )

    task_tuple, meta_tuple = train_test_split(
        unbiased,
        train_percentage=args.test_pcnt / (args.test_pcnt + args.pretrain_pcnt),
        random_seed=args.data_split_seed,
    )
    return Triplet(meta=meta_tuple, task=task_tuple, task_train=task_train_tuple)


def get_data_tuples(*pytorch_datasets):
    """Convert pytorch datasets to datatuples"""
    # FIXME: this is needed because the information about feature names got lost
    sens_attrs = Adult().feature_split["s"]
    return (pytorch_data_to_dataframe(data, sens_attrs=sens_attrs) for data in pytorch_datasets)


def pytorch_data_to_dataframe(dataset, sens_attrs=None):
    """Load a pytorch dataset into a DataTuple consisting of Pandas DataFrames

    Args:
        dataset: PyTorch dataset
        sens_attrs: (optional) list of names of the sensitive attributes
    """
    # create data loader with one giant batch
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # get the data
    data = next(iter(data_loader))
    # convert it to Pandas DataFrames
    data = [pd.DataFrame(tensor.detach().cpu().numpy()) for tensor in data]
    if sens_attrs:
        data[1].columns = sens_attrs
    # create a DataTuple
    return DataTuple(x=data[0], s=data[1], y=data[2])


def shuffle_s(dt: DataTuple) -> DataTuple:
    return dt.replace(s=shuffle_df(dt.s, random_state=42))


def shuffle_y(dt: DataTuple) -> DataTuple:
    return dt.replace(y=shuffle_df(dt.y, random_state=42))
