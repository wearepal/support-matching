from lapjv import lapjv
import numpy as np
from numpy.testing import assert_array_equal


def test_validate() -> None:
    # prepare
    counts = np.array([[51, 92, 14, 71], [60, 20, 82, 86], [74, 74, 87, 99], [23, 2, 21, 52]])

    # second method
    row_ind, col_ind, other = lapjv(-counts)
    assert other[0] == -300
    assert_array_equal(row_ind, np.array([1, 2, 0, 3]))
    assert_array_equal(col_ind, np.array([2, 0, 1, 3]))
