from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

import torch.nn as nn

from src.arch.common import Activation, BiaslessLayerNorm

from .base import PredictorFactory, PredictorFactoryOut

__all__ = ["Fcn"]


class NormType(Enum):
    BN = partial(nn.BatchNorm1d)
    LN = partial(BiaslessLayerNorm)


@dataclass
class Fcn(PredictorFactory):
    num_hidden: int = 0
    hidden_dim: Optional[int] = None
    activation: Activation = Activation.GELU
    norm: Optional[NormType] = NormType.LN
    dropout_prob: float = 0.0
    final_bias: bool = True

    def _make_block(self, in_features: int, *, out_features: int) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if self.norm is not None:
            block.append(self.norm.value(out_features))
        block.append(self.activation.value())
        if self.dropout_prob > 0:
            block.append(nn.Dropout(p=self.dropout_prob))
        return block

    def __call__(self, input_dim: int, *, target_dim: int) -> PredictorFactoryOut[nn.Sequential]:
        predictor = nn.Sequential(nn.Flatten())
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
