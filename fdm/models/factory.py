from typing import Dict, Tuple, Union

from fdm.models import Classifier
from fdm.models.configs import ModelFn

__all__ = ["build_discriminator"]


def build_discriminator(
    input_shape: Tuple[int, ...],
    target_dim: int,
    model_fn: ModelFn,
    model_kwargs: Dict[str, Union[float, str, bool]],
    optimizer_kwargs=None,
):
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, target_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )

    return discriminator
