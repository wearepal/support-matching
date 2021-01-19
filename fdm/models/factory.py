from typing import Optional, Tuple

from fdm.models import Classifier
from shared.utils import ModelFn

__all__ = ["build_discriminator"]


def build_discriminator(
    input_shape: Tuple[int, ...],
    target_dim: int,
    model_fn: ModelFn,
    optimizer_kwargs: Optional[dict] = None,
) -> Classifier:
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    return Classifier(
        model_fn(in_dim, target_dim),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )
