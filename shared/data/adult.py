"""Definition of the Adult dataset"""
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from ethicml.data import adult, load_data, Dataset
from ethicml.preprocessing import (
    BalancedTestSplit,
    DataSplitter,
    ProportionalSplit,
)
from ethicml.utility import DataTuple
from shared.configs import BaseArgs
from ethicml.preprocessing.domain_adaptation import make_valid_variable_name, query_dt

from .dataset_wrappers import DataTupleDataset

__all__ = ["get_data_tuples", "load_adult_data", "pytorch_data_to_dataframe"]

ADULT_DATASET: Dataset = None  # type: ignore[assignment]


class DataTupleTriplet(NamedTuple):
    """Small helper class; basically for enabling named returns"""

    context: DataTuple
    test: DataTuple
    train: DataTuple


def _random_split(data: DataTuple, first_pcnt: float, seed: int) -> Tuple[DataTuple, DataTuple]:
    if len(data) == 0:
        return data, data
    splitter = ProportionalSplit(train_percentage=first_pcnt, start_seed=seed)
    return splitter(data)[0:2]


def get_invisible_demographics(
    data: DataTuple, unbiased_pcnt: float, seed: int = 42, data_efficient: bool = True,
) -> Tuple[DataTuple, DataTuple]:
    """Split the given data into a biased subset and a normal subset.

    The two subsets don't generally sum up to the whole set.

    Args:
        data: data in form of a DataTuple
        unbiased_pcnt: how much of the data should be reserved for the unbiased subset
        seed: random seed for the splitting
        data_efficient: if True, try to keep as many data points as possible

    Returns:
        biased and unbiased dataset
    """
    assert 0 <= unbiased_pcnt <= 1, f"unbiased_pcnt: {unbiased_pcnt}"
    s_name = data.s.columns[0]
    y_name = data.y.columns[0]
    s_name = make_valid_variable_name(s_name)
    y_name = make_valid_variable_name(y_name)
    s_values = np.unique(data.s.to_numpy())
    y_values = np.unique(data.y.to_numpy())
    s_0, s_1 = s_values
    y_0, y_1 = s_values

    normal_subset, for_biased_subset = _random_split(data, first_pcnt=unbiased_pcnt, seed=seed)

    # two groups are missing (all males or all females)
    # one_s_only = query_dt(
    #    for_biased_subset, f"({s_name} == {s_1})"
    # )
    # one group is missing
    one_s_only = query_dt(
        for_biased_subset,
        f"({s_name} == {s_0} & {y_name} == {y_0}) | ({s_name} == {s_1} & {y_name} == {y_0}) | ({s_name} == {s_1} & {y_name} == {y_1})",
    )
    print("ensuring that only one group is missing")

    one_s_only = one_s_only.replace(name=f"{data.name})")
    normal_subset = normal_subset.replace(name=f"{data.name})")
    return one_s_only, normal_subset


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
        train_tuple, unbiased = get_invisible_demographics(  # get_biased_subset(
            data=data,
            # mixing_factor=args.mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.context_pcnt,
            seed=args.data_split_seed,
            data_efficient=True,
        )
        if args.balanced_context:
            make_balanced = BalancedTestSplit(train_percentage=0, start_seed=args.data_split_seed)
            _, unbiased, _ = make_balanced(unbiased)
    else:
        train_tuple, unbiased, _ = BalancedTestSplit(
            train_percentage=1 - args.test_pcnt - args.context_pcnt,
            start_seed=args.data_split_seed,
        )(data)

    context_pcnt = args.context_pcnt / (args.test_pcnt + args.context_pcnt)
    context_splitter: DataSplitter
    # if balanced_test is True and and `unbiased` has not been made balanced before...
    if args.balanced_test and args.biased_train and not args.balanced_context:
        context_splitter = BalancedTestSplit(
            train_percentage=context_pcnt, start_seed=args.data_split_seed
        )
    else:
        context_splitter = ProportionalSplit(
            train_percentage=context_pcnt, start_seed=args.data_split_seed
        )
    context_tuple, test_tuple, _ = context_splitter(unbiased)
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
