from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Optional, Type, TypeVar
from typing_extensions import override

from ranzen.torch import DcModule
from torch import Tensor
import torch.nn as nn

from src.arch.aggregation import BatchAggregator, GatedAggregator, KvqAggregator
from src.arch.common import Activation, BiaslessLayerNorm

from .base import PredictorFactory, PredictorFactoryOut

__all__ = ["BatchAggregatorEnum", "Fcn", "NormType", "SetFcn", "SetPredictor"]


class NormType(Enum):
    BN = (nn.BatchNorm1d,)
    LN = (BiaslessLayerNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init


@dataclass
class Fcn(PredictorFactory):
    """Fully connected network."""

    num_hidden: int = 0
    hidden_dim: Optional[int] = None
    activation: Activation = Activation.GELU
    norm: Optional[NormType] = NormType.LN
    dropout_prob: float = 0.0
    final_bias: bool = True
    input_norm: bool = True

    def _make_block(self, in_features: int, *, out_features: int) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if self.norm is not None:
            block.append(self.norm.init(out_features))
        block.append(self.activation.init())
        if self.dropout_prob > 0:
            block.append(nn.Dropout(p=self.dropout_prob))
        return block

    @override
    def __call__(
        self, input_dim: int, *, target_dim: int, batch_size: Optional[int] = None
    ) -> PredictorFactoryOut[nn.Sequential]:
        predictor = nn.Sequential(nn.Flatten())
        if self.input_norm and (self.norm is not None):
            predictor.append(self.norm.init(input_dim))
        curr_dim = input_dim
        if self.num_hidden > 0:
            hidden_dim = input_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(self._make_block(in_features=curr_dim, out_features=hidden_dim))
                curr_dim = hidden_dim
        predictor.append(
            nn.Linear(in_features=curr_dim, out_features=target_dim, bias=self.final_bias)
        )
        return predictor, target_dim


A = TypeVar("A", bound=BatchAggregator)


@dataclass(repr=False, eq=False)
class SetPredictor(DcModule, Generic[A]):
    pre: nn.Module
    agg: A
    post: nn.Module

    @property
    def batch_size(self) -> int:
        return self.agg.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.agg.batch_size = value

    def forward(self, x: Tensor, *, batch_size: Optional[int] = None) -> Tensor:  # type: ignore
        if batch_size is not None:
            self.batch_size = batch_size
        return self.post(self.agg(self.pre(x)))


class BatchAggregatorEnum(Enum):
    KVQ = (KvqAggregator,)
    GATED = (GatedAggregator,)

    def __init__(self, init: Type[BatchAggregator]) -> None:
        self.init = init


@dataclass(eq=False)
class SetFcn(PredictorFactory):
    hidden_dim_pre: Optional[int] = None
    hidden_dim_post: Optional[int] = None
    num_hidden_pre: int = 1
    num_hidden_post: int = 1
    agg_input_dim: Optional[int] = None
    activation: Activation = Activation.GELU
    norm: NormType = NormType.LN
    dropout_prob: float = 0
    final_bias: bool = True
    input_norm: bool = True
    num_heads: int = 1
    head_dim: Optional[int] = 512
    num_blocks: int = 0
    mean_query: bool = True

    def _pre_agg_fcn(self, input_dim: int) -> PredictorFactoryOut[nn.Sequential]:
        agg_input_dim = self.agg_input_dim
        if agg_input_dim is None:
            agg_input_dim = input_dim if self.hidden_dim_pre is None else self.hidden_dim_pre
        return Fcn(
            num_hidden=self.num_hidden_pre,
            hidden_dim=self.hidden_dim_pre,
            activation=self.activation,
            norm=self.norm,
            final_bias=True,
            input_norm=self.input_norm,
        )(input_dim=input_dim, target_dim=agg_input_dim)

    def _post_agg_fcn(
        self, input_dim: int, *, target_dim: int
    ) -> PredictorFactoryOut[nn.Sequential]:
        return Fcn(
            num_hidden=self.num_hidden_pre,
            hidden_dim=self.hidden_dim_pre,
            activation=self.activation,
            norm=self.norm,
            final_bias=self.final_bias,
        )(input_dim, target_dim=target_dim)

    def _aggregator(
        self, input_dim: int, *, batch_size: int
    ) -> PredictorFactoryOut[BatchAggregator]:
        return (
            KvqAggregator(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                dim=input_dim,
                head_dim=self.head_dim,
                mean_query=self.mean_query,
            ),
            input_dim,
        )

    @override
    def __call__(
        self, input_dim: int, *, target_dim: int, batch_size: int
    ) -> PredictorFactoryOut[SetPredictor[BatchAggregator]]:
        fcn_pre, input_dim = self._pre_agg_fcn(input_dim=input_dim)
        aggregator, input_dim = self._aggregator(
            input_dim=input_dim,
            batch_size=batch_size,
        )
        fcn_post, target_dim = self._post_agg_fcn(
            input_dim=input_dim,
            target_dim=target_dim,
        )
        model = nn.Sequential(fcn_pre, aggregator, fcn_post)
        model = SetPredictor(pre=fcn_pre, agg=aggregator, post=fcn_post)
        return model, target_dim
