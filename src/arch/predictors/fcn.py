from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from ranzen import implements
import torch.nn as nn

from src.arch.aggregation import BatchAggregator, GatedAggregator, KvqAggregator
from src.arch.common import Activation, BiaslessLayerNorm

from .base import PredictorFactory, PredictorFactoryOut

__all__ = [
    "Fcn",
    "GatedSetFcn",
    "KvqSetFcn",
    "NormType",
    "SetFcn",
]


class NormType(Enum):
    BN = (nn.BatchNorm1d,)
    LN = (BiaslessLayerNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init


@dataclass
class Fcn(PredictorFactory):
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

    @implements(PredictorFactory)
    def __call__(
        self, input_dim: int, *, target_dim: int, **kwargs: Any
    ) -> PredictorFactoryOut[nn.Sequential]:
        predictor = nn.Sequential(nn.Flatten())
        if self.input_norm and (self.norm is not None):
            predictor.append(self.norm.init(input_dim))
        curr_dim = input_dim
        if self.num_hidden > 0:
            hidden_dim = input_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(
                    self._make_block(
                        in_features=curr_dim,
                        out_features=hidden_dim,
                    )
                )
                curr_dim = hidden_dim
        predictor.append(
            nn.Linear(
                in_features=curr_dim,
                out_features=target_dim,
                bias=self.final_bias,
            )
        )
        return predictor, target_dim


@dataclass
class SetFcn(PredictorFactory):
    hidden_dim_pre: Optional[int]
    hidden_dim_post: Optional[int]
    num_hidden_pre: int = 2
    num_hidden_post: int = 2
    agg_input_dim: Optional[int] = None
    activation: Activation = Activation.GELU
    norm: NormType = NormType.LN
    dropout_prob: float = 0
    final_bias: bool = True
    input_norm: bool = True

    def _pre_agg_fcn(self, input_dim: int) -> PredictorFactoryOut[nn.Sequential]:
        agg_input_dim = self.agg_input_dim
        if agg_input_dim is None:
            if self.hidden_dim_pre is None:
                agg_input_dim = input_dim
            else:
                agg_input_dim = self.hidden_dim_pre
        return Fcn(
            num_hidden=self.num_hidden_pre,
            hidden_dim=self.hidden_dim_pre,
            activation=self.activation,
            norm=self.norm,
            final_bias=True,
            input_norm=self.input_norm,
        )(
            input_dim=input_dim,
            target_dim=agg_input_dim,
        )

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

    @abstractmethod
    def _aggregator(
        self, input_dim: int, *, batch_size: int
    ) -> PredictorFactoryOut[BatchAggregator]:
        ...

    @implements(PredictorFactory)
    def __call__(
        self,
        input_dim: int,
        *,
        target_dim: int,
        batch_size: int,
        **kwargs: Any,
    ) -> PredictorFactoryOut[nn.Sequential]:
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
        return model, target_dim


@dataclass
class GatedSetFcn(SetFcn):
    @implements(SetFcn)
    def _aggregator(
        self, input_dim: int, *, batch_size: int
    ) -> PredictorFactoryOut[BatchAggregator]:
        return (
            GatedAggregator(
                batch_size=batch_size,
                dim=input_dim,
            ),
            input_dim,
        )


@dataclass
class KvqSetFcn(SetFcn):
    num_attn_heads: int = 1

    @implements(SetFcn)
    def _aggregator(
        self, input_dim: int, *, batch_size: int
    ) -> PredictorFactoryOut[BatchAggregator]:
        return (
            KvqAggregator(
                batch_size=batch_size,
                dim=input_dim,
                num_heads=self.num_attn_heads,
            ),
            input_dim,
        )
