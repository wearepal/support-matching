import torch

from fdm.optimisation.utils import get_all_num_samples, weight_for_balance, weights_with_counts
from shared.utils import label_to_class_id


def test_weight_for_balance() -> None:
    cluster_ids = torch.tensor([4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 0])
    weights, n_clusters, min_count, max_count = weight_for_balance(cluster_ids)

    torch.testing.assert_allclose(
        weights,
        torch.tensor([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0]),
        rtol=0,
        atol=0,
    )
    assert n_clusters == 4
    assert min_count == 1
    assert max_count == 4


def test_get_all_num_samples() -> None:
    s_dim = 2
    s = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1])
    y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    class_ids = label_to_class_id(s=s, y=y, s_count=s_dim)
    _, y_w_and_c = weights_with_counts(y)
    _, quad_w_and_c = weights_with_counts(class_ids)

    all_num_samples = get_all_num_samples(
        quad_w_and_c=quad_w_and_c, y_w_and_c=y_w_and_c, s_dim=s_dim
    )
    expected = [2 * 4, 4 * 4, 3 * 2]
    # check that the two lists are identical (except for the order)
    assert len(expected) == len(all_num_samples)
    assert all(num_samples in all_num_samples for num_samples in expected)
