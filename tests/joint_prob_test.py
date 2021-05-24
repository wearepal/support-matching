from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.testing import assert_allclose

from shared.utils import get_joint_probability, label_to_class_id


def test_compatibility_binary() -> None:
    s = torch.tensor([0, 1, 0, 1, 1, 0])
    y = torch.tensor([1, 1, 0, 0, 0, 1])
    _compare(s, y, 2, 2)


def test_compatibility_multiclass() -> None:
    s = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    _compare(s, y, 3, 4)


def _compare(s: Tensor, y: Tensor, s_count: int, y_count: int) -> None:
    """Check wether label_to_class_id() and get_joint_probability() give the same answer."""
    class_ids = label_to_class_id(s=s, y=y, s_count=s_count)

    s_probs: Tensor = F.one_hot(s, num_classes=s_count).float()
    y_probs: Tensor = F.one_hot(y, num_classes=y_count).float()
    joint = get_joint_probability(s_probs=s_probs, y_probs=y_probs)

    assert_allclose(class_ids, joint.argmax(dim=-1), atol=0, rtol=0)


def test_compatibility_reversed() -> None:
    s_probs = torch.tensor([[0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.4, 0.5]])
    y_probs = torch.tensor([[0.4, 0.5], [0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])

    joint = get_joint_probability(s_probs=s_probs, y_probs=y_probs)

    class_ids = label_to_class_id(s=s_probs.argmax(dim=-1), y=y_probs.argmax(dim=-1), s_count=2)

    assert_allclose(class_ids, joint.argmax(dim=-1), atol=0, rtol=0)
