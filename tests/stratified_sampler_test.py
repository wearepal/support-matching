from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.utils import StratifiedSampler


def count_true(mask: np.ndarray[np.bool_]) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]


@pytest.fixture
def group_ids() -> list[int]:
    return torch.cat(
        [torch.full((100,), 0), torch.full((200,), 1), torch.full((400,), 2), torch.full((800,), 3)]
    ).tolist()


def test_simple(group_ids: list[int]):
    num_samples_per_group = 800
    indexes = np.fromiter(
        iter(
            StratifiedSampler(group_ids, num_samples_per_group, replacement=True, multipliers=None)
        ),
        np.int64,
    )
    assert len(indexes) == 4 * num_samples_per_group
    assert count_true(indexes < 100) == num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == num_samples_per_group
    assert count_true((300 <= indexes) & (indexes < 700)) == num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group

    # confirm that the interleaving works correctly
    assert indexes[0] < 100
    assert 100 <= indexes[1] < 300
    assert 300 <= indexes[2] < 700
    assert 700 <= indexes[3] < 1500
    assert indexes[4] < 100


def test_without_replacement(group_ids: list[int]):
    num_samples_per_group = 100
    indexes = np.fromiter(
        iter(
            StratifiedSampler(group_ids, num_samples_per_group, replacement=False, multipliers=None)
        ),
        np.int64,
    )
    assert len(indexes) == 4 * num_samples_per_group
    assert count_true(indexes < 100) == num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == num_samples_per_group
    assert count_true((300 <= indexes) & (indexes < 700)) == num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group

    # confirm that the interleaving works correctly
    assert indexes[0] < 100
    assert 100 <= indexes[1] < 300
    assert 300 <= indexes[2] < 700
    assert 700 <= indexes[3] < 1500
    assert indexes[4] < 100


def test_with_multipliers(group_ids: list[int]):
    num_samples_per_group = 800
    indexes = np.fromiter(
        iter(
            StratifiedSampler(
                group_ids, num_samples_per_group, replacement=True, multipliers={0: 2, 1: 0, 2: 3}
            )
        ),
        np.int64,
    )
    assert len(indexes) == (2 + 0 + 3 + 1) * num_samples_per_group
    assert count_true(indexes < 100) == 2 * num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == 0
    assert count_true((300 <= indexes) & (indexes < 700)) == 3 * num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group

    # confirm that the interleaving works correctly
    assert indexes[0] < 100
    assert indexes[1] < 100
    assert 300 <= indexes[2] < 700
    assert 300 <= indexes[3] < 700
    assert 300 <= indexes[4] < 700
    assert 700 <= indexes[5] < 1500
    assert indexes[6] < 100
