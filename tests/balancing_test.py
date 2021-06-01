import torch

from suds.optimisation.utils import _get_multipliers_and_group_size


def test_binary_s_and_y() -> None:
    class_ids = torch.tensor([0, 0, 0, 0, 1, 1, 3, 3, 3, 3, 3])
    s_count = 2
    multipliers, group_sizes = _get_multipliers_and_group_size(class_ids, s_count)

    assert multipliers == {0: 1, 1: 1, 3: 2}
    assert group_sizes == [2, 2, 4]


def test_multiclass_s_and_y() -> None:
    # setup:
    #
    # y=0
    #   s=0: 0
    #   s=1: 1
    # y=1
    #   s=0: 3
    #   s=1: 4
    #   s=2: 5
    # y=2
    #   s=2: 8
    #
    # this means that 8 gets a multiplier of 6, 3/4/5 get 2 and 0/1 get 3
    class_ids = torch.tensor([0, 0, 0, 1, 1, 1, 3, 3, 4, 4, 5, 5, 8, 8, 8, 8, 8, 8])
    s_count = 3
    multipliers, group_sizes = _get_multipliers_and_group_size(class_ids, s_count)

    assert multipliers == {0: 3, 1: 3, 3: 2, 4: 2, 5: 2, 8: 6}
    assert group_sizes == [1, 1, 1, 1, 1, 1]
