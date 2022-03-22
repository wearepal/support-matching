from __future__ import annotations

from shared.models.configs.classifiers import ModelFactory

from .classifier import Classifier

__all__ = ["build_classifier"]


def build_classifier(
    input_shape: tuple[int, ...],
    *,
    target_dim: int,
    model_fn: ModelFactory,
    optimizer_kwargs: dict | None = None,
) -> Classifier:
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    return Classifier(
        model_fn(in_dim, target_dim), num_classes=num_classes, optimizer_kwargs=optimizer_kwargs
    )
