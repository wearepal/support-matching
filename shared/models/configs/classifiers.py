from typing import List, Optional, Sequence, Union

from torch import nn

from shared.layers import Aggregator
from shared.utils import ModelFn, prod

__all__ = ["Mp32x23Net", "Mp64x64Net", "FcNet", "ModelAggregatorWrapper"]


class Mp32x23Net:
    def __init__(self, batch_norm: bool):
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

    def __call__(self, input_dim: int, target_dim: int) -> nn.Sequential:
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


class FcNet:
    def __init__(self, hidden_dims: Optional[Sequence[int]]):
        self.hidden_dims = hidden_dims

    @staticmethod
    def _fc_block(in_dim: int, out_dim: int) -> List[nn.Module]:
        _block: List[nn.Module] = []
        _block += [nn.Linear(in_dim, out_dim)]
        _block += [nn.SELU()]
        return _block

    def __call__(self, input_dim: Union[int, Sequence[int]], target_dim: int) -> nn.Sequential:
        hidden_dims = self.hidden_dims or []

        layers: List[nn.Module] = [nn.Flatten()]
        if not isinstance(input_dim, int):
            input_dim = prod(input_dim)

        for output_dim in hidden_dims:
            layers.extend(self._fc_block(input_dim, output_dim))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, target_dim))

        return nn.Sequential(*layers)


class Mp64x64Net:
    def __init__(self, batch_norm: bool):
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

    def __call__(self, input_dim: int, target_dim: int) -> nn.Sequential:
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


class ModelAggregatorWrapper:
    def __init__(self, model_fn: ModelFn, aggregator: Aggregator, embed_dim: int):
        self.model_fn = model_fn
        self.aggregator = aggregator
        self.embed_dim = embed_dim

    def __call__(self, input_dim: int, target_dim: int) -> nn.Module:
        assert target_dim == self.aggregator.output_dim
        return nn.Sequential(self.model_fn(input_dim, self.embed_dim), self.aggregator)
