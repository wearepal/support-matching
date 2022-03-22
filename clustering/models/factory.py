from typing import Dict, Optional, Tuple

from clustering.models import Classifier
from shared.models.factory import ModelFactory

__all__ = ["build_classifier"]


def build_classifier(
    input_shape: Tuple[int, ...],
    *,
    target_dim: int,
    model_fn: ModelFactory,
    optimizer_kwargs: Optional[Dict[str, float]] = None,
) -> Classifier:
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    return Classifier(
        model_fn(in_dim, target_dim=target_dim),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )
