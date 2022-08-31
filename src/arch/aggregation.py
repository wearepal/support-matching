"""Modules that aggregate over a batch."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from ranzen import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from src.arch.common import Activation

__all__ = [
    "BatchAggregator",
    "GatedAggregator",
    "KvqAggregator",
]


@dataclass(eq=False)
class BatchAggregator(DcModule):
    dim: int
    batch_size: int

    def batch_to_bags(self, batch: Tensor) -> Tensor:
        """
        Reshape a batch so that it's a batch of bags.

        This is the only certified way of producing bags. Use all other methods at your own risk.
        """
        return batch.view(-1, *batch.shape[1:], self.batch_size).movedim(-1, 0)


@dataclass(eq=False)
class KvqAggregator(BatchAggregator):

    activation: Activation = Activation.GELU
    dropout: float = 0.0

    attn: nn.MultiheadAttention = field(init=False)
    act_fn: Callable[[Tensor], Tensor] = field(init=False)
    num_heads: int = 1

    def __post_init__(self) -> None:
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=True,
        )
        self.act_fn = self.activation.value()

    @implements(BatchAggregator)
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        # `attn` expects the *second* dimension to be the batch dimension
        # that's why we have to transpose here
        inputs_batched = self.batch_to_bags(inputs).transpose(0, 1)  # shape: (bag, batch, latent)
        # for the query we just use an average of the bags
        query = inputs_batched.mean(dim=0, keepdim=True)
        key = inputs_batched
        value = key
        output = self.act_fn(self.attn(query=query, key=key, value=value, need_weights=False)[0])
        return output.view(-1, self.dim)


@dataclass(eq=False)
class GatedAggregator(BatchAggregator):

    v: Parameter = field(init=False)
    u: Parameter = field(init=False)
    w: Parameter = field(init=False)

    attention_weights: Optional[Tensor] = field(init=False, default=None)

    def __post_init__(self):
        self.v = Parameter(torch.empty(self.dim, self.dim), requires_grad=True)
        self.u = Parameter(torch.empty(self.dim, self.dim), requires_grad=True)
        self.w = Parameter(torch.empty(1, self.dim), requires_grad=True)
        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.w)

    @implements(BatchAggregator)
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        inputs = inputs.flatten(start_dim=1)
        logits = torch.tanh(inputs @ self.v.t()) * torch.sigmoid(inputs @ self.u.t()) @ self.w.t()
        logits_batched = self.batch_to_bags(logits)
        weights = logits_batched.softmax(dim=1)
        self.attention_weights = weights.squeeze(-1).detach().cpu()
        inputs_batched = self.batch_to_bags(inputs)
        return torch.sum(weights * inputs_batched, dim=1, keepdim=False)
