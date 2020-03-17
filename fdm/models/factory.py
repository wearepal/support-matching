from typing import Dict, Tuple, Union

from fdm.models import Classifier
from fdm.models.configs import ModelFn
from fdm.utils import product

__all__ = ["build_discriminator"]


def build_discriminator(
    input_shape: Tuple[int, ...],
    target_dim: int,
    train_on_recon: bool,
    frac_enc: float,
    model_fn: ModelFn,
    model_kwargs: Dict[str, Union[float, str, bool]],
    optimizer_kwargs=None,
):
    in_dim = input_shape[0]

    # this is done in models/inn.py
    if not train_on_recon and len(input_shape) > 2:
        in_dim = round(frac_enc * int(product(input_shape)))

    num_classes = target_dim if target_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, target_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )

    return discriminator
