from collections.abc import Iterable
from typing import Union

from torch.nn.parameter import Parameter

__all__ = ["exclude_from_weight_decay"]


def exclude_from_weight_decay(
    named_params: Iterable[tuple[str, Parameter]],
    weight_decay: float = 0.0,
    exclusion_patterns: tuple[str, ...] = ("bias",),
) -> list[dict[str, Union[list[Parameter], float]]]:
    params: list[Parameter] = []
    excluded_params: list[Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in exclusion_patterns):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {
            "params": excluded_params,
            "weight_decay": 0.0,
        },
    ]
