"""Definition of the Adult dataset"""
import logging
from typing import List, NamedTuple, Tuple

import ethicml as em
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from shared.configs import BaseArgs

from .dataset_wrappers import DataTupleDataset

__all__ = ["get_data_tuples", "load_adult_data", "pytorch_data_to_dataframe"]

log = logging.getLogger(__name__.split(".")[-1].upper())

ADULT_DATASET: em.Dataset = None  # type: ignore[assignment]
SENS_ATTRS: List[str] = []


class DataTupleTriplet(NamedTuple):
    """Small helper class; basically for enabling named returns"""

    context: em.DataTuple
    test: em.DataTuple
    train: em.DataTuple


def _random_split(
    data: em.DataTuple, first_pcnt: float, seed: int
) -> Tuple[em.DataTuple, em.DataTuple]:
    if len(data) == 0:
        return data, data
    splitter = em.ProportionalSplit(train_percentage=first_pcnt, start_seed=seed)
    return splitter(data)[0:2]


def get_invisible_demographics(
    data: em.DataTuple, unbiased_pcnt: float, seed: int, missing_s: List[int]
) -> Tuple[em.DataTuple, em.DataTuple]:
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
    s = np.unique(data.s.to_numpy())
    y = np.unique(data.y.to_numpy())

    normal_subset, for_biased_subset = _random_split(data, first_pcnt=unbiased_pcnt, seed=seed)

    # two groups are missing (all males or all females)
    # one_s_only = query_dt(
    #    for_biased_subset, f"({s_name} == {s_1})"
    # )
    # one group is missing
    if missing_s:
        if len(missing_s) == 1 and missing_s[0] == 0:
            query = (
                f"(`{s_name}` == {s[1]} & `{y_name}` == {y[0]})"
                f" | (`{s_name}` == {s[1]} & `{y_name}` == {y[1]})"
            )
            log.info("removing s=0")
        else:
            raise ValueError(f"Unsupported missing group {missing_s}")
    else:
        query = (
            f"(`{s_name}` == {s[0]} & `{y_name}` == {y[0]})"
            f" | (`{s_name}` == {s[1]} & `{y_name}` == {y[0]})"
            f" | (`{s_name}` == {s[1]} & `{y_name}` == {y[1]})"
        )
        log.info("ensuring that only one group is missing")
    one_s_only = em.query_dt(for_biased_subset, query)

    one_s_only = one_s_only.replace(name=f"{data.name})")
    normal_subset = normal_subset.replace(name=f"{data.name})")
    return one_s_only, normal_subset


def load_adult_data(cfg: BaseArgs) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    global ADULT_DATASET
    ADULT_DATASET = em.adult(
        split=cfg.data.adult_split.name, binarize_nationality=cfg.data.drop_native
    )
    data = ADULT_DATASET.load(ordered=True)
    global SENS_ATTRS
    SENS_ATTRS = data.s.columns

    disc_feature_groups = ADULT_DATASET.disc_feature_groups
    assert disc_feature_groups is not None
    cont_feats = ADULT_DATASET.continuous_features

    tuples: DataTupleTriplet = biased_split(cfg, data)
    context, test, train = tuples.context, tuples.test, tuples.train

    scaler = StandardScaler()

    train_x = train.x
    train_x[cont_feats] = scaler.fit_transform(train.x[cont_feats].to_numpy(np.float32))
    test_x = test.x
    test_x[cont_feats] = scaler.transform(test.x[cont_feats].to_numpy(np.float32))
    context_x = context.x
    context_x[cont_feats] = scaler.transform(context.x[cont_feats].to_numpy(np.float32))

    if cfg.data.drop_discrete:
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


def biased_split(cfg: BaseArgs, data: em.DataTuple) -> DataTupleTriplet:
    """Split the dataset such that the training set is biased."""
    if cfg.bias.adult_biased_train:
        train_tuple, unbiased = get_invisible_demographics(  # get_biased_subset(
            data=data,
            # mixing_factor=args.mixing_factor,
            unbiased_pcnt=cfg.data.test_pcnt + cfg.data.context_pcnt,
            seed=cfg.misc.data_split_seed,
            missing_s=cfg.bias.missing_s,
        )
    else:
        train_tuple, unbiased, _ = em.BalancedTestSplit(
            train_percentage=1 - cfg.data.test_pcnt - cfg.data.context_pcnt,
            start_seed=cfg.misc.data_split_seed,
        )(data)

    context_pcnt = cfg.data.context_pcnt / (cfg.data.test_pcnt + cfg.data.context_pcnt)
    context_splitter: em.DataSplitter

    if cfg.data.adult_balanced_test and cfg.bias.adult_biased_train:
        context_splitter = em.BalancedTestSplit(
            train_percentage=context_pcnt,
            start_seed=cfg.misc.data_split_seed,
            balance_type="P(s,y)=0.25" if cfg.data.balance_all_quadrants else "P(s|y)=0.5",
        )
    else:
        context_splitter = em.ProportionalSplit(
            train_percentage=context_pcnt, start_seed=cfg.misc.data_split_seed
        )
    context_tuple, test_tuple, _ = context_splitter(unbiased)
    return DataTupleTriplet(context=context_tuple, test=test_tuple, train=train_tuple)


def get_data_tuples(*pytorch_datasets):
    """Convert pytorch datasets to datatuples"""
    return (pytorch_data_to_dataframe(data, sens_attrs=SENS_ATTRS) for data in pytorch_datasets)


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
    if sens_attrs is not None:
        data[1].columns = sens_attrs
    # create a DataTuple
    return em.DataTuple(x=data[0], s=data[1], y=data[2])
