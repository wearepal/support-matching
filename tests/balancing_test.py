# import torch

# from fdm.optimisation.utils import get_all_num_samples, weight_for_balance, weights_with_counts
# from shared.utils import label_to_class_id


# def test_weight_for_balance() -> None:
#     cluster_ids = torch.tensor([4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 0])
#     weights, n_clusters, min_count, max_count = weight_for_balance(cluster_ids)

#     torch.testing.assert_allclose(
#         weights,
#         torch.tensor([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0]),
#         rtol=0,
#         atol=0,
#     )
#     assert n_clusters == 4
#     assert min_count == 1
#     assert max_count == 4
