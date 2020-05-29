"""Definition of the Adult dataset"""
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ethicml.data import adult, load_data, Dataset
from ethicml.preprocessing import (
    BalancedTestSplit,
    ProportionalSplit,
    get_biased_subset,
)
from ethicml.utility import DataTuple
from shared.configs import BaseArgs

from .dataset_wrappers import DataTupleDataset

__all__ = ["get_data_tuples", "load_adult_data", "pytorch_data_to_dataframe"]

ADULT_DATASET: Dataset = None  # type: ignore[assignment]


class DataTupleTriplet(NamedTuple):
    """Small helper class; basically for enabling named returns"""

    context: DataTuple
    test: DataTuple
    train: DataTuple


def load_adult_data(args: BaseArgs) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    global ADULT_DATASET
    ADULT_DATASET = adult(binarize_nationality=args.drop_native)
    data = load_data(ADULT_DATASET, ordered=True, generate_dummies=True)

    disc_feature_groups = ADULT_DATASET.disc_feature_groups
    assert disc_feature_groups is not None
    cont_feats = ADULT_DATASET.continuous_features

    tuples: DataTupleTriplet = biased_split(args, data)
    context, test, train = tuples.context, tuples.test, tuples.train

    scaler = StandardScaler()

    train_x = train.x
    train_x[cont_feats] = scaler.fit_transform(train.x[cont_feats].to_numpy(np.float32))
    test_x = test.x
    test_x[cont_feats] = scaler.transform(test.x[cont_feats].to_numpy(np.float32))
    context_x = context.x
    context_x[cont_feats] = scaler.transform(context.x[cont_feats].to_numpy(np.float32))

    if args.drop_discrete:
        context_x = context_x[cont_feats]
        train_x = train_x[cont_feats]
        test_x = test_x[cont_feats]
        disc_feature_groups = {}

    train = train.replace(x=train_x)
    test = test.replace(x=test_x)
    context = context.replace(x=context_x)

    cont_features = ADULT_DATASET.continuous_features
    context_dataset = DataTupleDataset(
        context, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    train_dataset = DataTupleDataset(
        train, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    test_dataset = DataTupleDataset(
        test, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )
    return context_dataset, train_dataset, test_dataset


def biased_split(args: BaseArgs, data: DataTuple) -> DataTupleTriplet:
    """Split the dataset such that the training set is biased."""
    if args.biased_train:
        train_tuple, unbiased = get_biased_subset(
            data=data,
            mixing_factor=args.mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.context_pcnt,
            seed=args.data_split_seed,
            data_efficient=True,
        )
    else:
        train_tuple, unbiased, _ = BalancedTestSplit(
            train_percentage=1 - args.test_pcnt - args.context_pcnt,
            start_seed=args.data_split_seed,
        )(data)

    test_tuple, context_tuple, _ = ProportionalSplit(
        train_percentage=args.test_pcnt / (args.test_pcnt + args.context_pcnt),
        start_seed=args.data_split_seed,
    )(unbiased)
    return DataTupleTriplet(context=context_tuple, test=test_tuple, train=train_tuple)


def get_data_tuples(*pytorch_datasets):
    """Convert pytorch datasets to datatuples"""
    sens_attrs = ADULT_DATASET.feature_split["s"]
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
