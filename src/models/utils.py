from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Union, final
from typing_extensions import Self

from torch import nn
from torch.nn.parameter import Parameter

__all__ = ["DcModule", "exclude_from_weight_decay"]


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
        {"params": excluded_params, "weight_decay": 0.0},
    ]


@dataclass(repr=False, eq=False)
class DcModule(nn.Module):
    @final
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj
