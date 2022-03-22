from typing import List, Optional, Sequence, TypeVar, Union
from typing_extensions import Protocol, runtime_checkable

from torch import nn

from shared.layers.aggregation import Aggregator
from shared.models.configs.classifiers import ModelFactory
from shared.utils import prod

__all__ = [
    "FcNet",
    "ModelAggregatorWrapper",
    "ModelFactory",
    "Mp32x23Net",
    "Mp64x64Net",
]


M_co = TypeVar("M_co", bound=nn.Module, covariant=True)


@runtime_checkable
class ModelFactory(Protocol[M_co]):
    def __call__(self, input_dim: int, *, target_dim: int) -> M_co:
        ...


class Mp32x23Net(ModelFactory[nn.Sequential]):
    def __init__(self, batch_norm: bool) -> None:
        self.batch_norm = batch_norm

    def _conv_block(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> List[nn.Module]:

        _block: List[nn.Module] = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    def __call__(self, input_dim: int, *, target_dim: int) -> nn.Sequential:
        layers = []
        layers.extend(self._conv_block(input_dim, 64, 5, 1, 0))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(64, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 256, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(256, 512, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(512, target_dim)]

        return nn.Sequential(*layers)


class FcNet(ModelFactory[nn.Sequential]):
    def __init__(
        self,
        hidden_dims: Optional[Sequence[int]],
        activation: nn.Module = nn.SELU(),
        final_layer_bias: bool = True,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.final_layer_bias = final_layer_bias

    def _fc_block(self, in_dim: int, out_dim: int) -> List[nn.Module]:
        _block: List[nn.Module] = []
        _block += [nn.Linear(in_dim, out_dim)]
        _block += [self.activation]
        return _block

    def __call__(self, input_dim: Union[int, Sequence[int]], *, target_dim: int) -> nn.Sequential:
        hidden_dims = self.hidden_dims or []

        layers: List[nn.Module] = [nn.Flatten()]
        if not isinstance(input_dim, int):
            input_dim = prod(input_dim)

        for output_dim in hidden_dims:
            layers.extend(self._fc_block(input_dim, output_dim))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, target_dim, bias=self.final_layer_bias))

        return nn.Sequential(*layers)


class Mp64x64Net(ModelFactory[nn.Sequential]):
    def __init__(self, batch_norm: bool) -> None:
        self.batch_norm = batch_norm

    def _conv_block(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> List[nn.Module]:
        _block: List[nn.Module] = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    def __call__(self, input_dim: int, *, target_dim: int) -> nn.Sequential:
        layers = []
        layers.extend(self._conv_block(input_dim, 64, 5, 1, 0))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(64, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 256, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(256, 512, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(512, target_dim)]

        return nn.Sequential(*layers)


class ModelAggregatorWrapper(ModelFactory[nn.Sequential]):
    def __init__(self, model_fn: ModelFactory, aggregator: Aggregator, input_dim: int) -> None:
        self.model_fn = model_fn
        self.aggregator = aggregator
        self.input_dim = input_dim

    def __call__(self, input_dim: int, *, target_dim: int) -> nn.Sequential:
        assert target_dim == self.aggregator.output_dim

        return nn.Sequential(
            self.model_fn(input_dim, target_dim=self.input_dim), nn.GELU(), self.aggregator
        )
