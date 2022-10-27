"""Modules that aggregate over a batch."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from ranzen import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from src.arch.common import BiaslessLayerNorm

__all__ = ["BatchAggregator", "GatedAggregator", "KvqAggregator", "BagMean"]


def batch_to_bags(batch: Tensor, *, batch_size: int) -> Tensor:
    """
    Reshape a batch so that it's a batch of bags.

    This is the only certified way of producing bags. Use all other methods at your own risk.
    """
    return batch.view(-1, *batch.shape[1:], batch_size).movedim(-1, 0)


@dataclass(eq=False)
class BatchAggregator(DcModule):
    batch_size: int

    def batch_to_bags(self, batch: Tensor) -> Tensor:
        """
        Reshape a batch so that it's a batch of bags.

        This is the only certified way of producing bags. Use all other methods at your own risk.
        """
        return batch_to_bags(batch=batch, batch_size=self.batch_size)


@dataclass(eq=False)
class BagMean(BatchAggregator):
    @implements(BatchAggregator)
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        inputs_batched = self.batch_to_bags(inputs)
        return inputs_batched.mean(1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            BiaslessLayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    @implements(nn.Module)
    def forward(self, x) -> Tensor:  # type: ignore
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        batch_size: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        mean_query: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.mean_query = mean_query
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=self.num_heads, dropout=self.dropout, bias=True
        )
        self.ln0 = BiaslessLayerNorm(self.dim)
        self.post_attn = nn.Linear(self.dim, self.dim)
        self.ff = FeedForward(dim=self.dim, hidden_dim=self.hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        inputs = self.ln0(inputs)
        # `attn` expects the *second* dimension to be the batch dimension
        # that's why we have to transpose here
        inputs_batched = batch_to_bags(inputs, batch_size=self.batch_size).transpose(
            0, 1
        )  # shape: (bag, batch, latent)
        if self.mean_query:
            query = inputs_batched.mean(0, keepdim=True)
        else:
            query = inputs_batched

        outputs, _ = self.attn(
            query=query, key=inputs_batched, value=inputs_batched, need_weights=False
        )
        if self.mean_query:
            outputs = outputs.movedim(0, 1).contiguous()
        outputs = outputs.view(-1, self.dim)
        outputs = self.post_attn(outputs)
        return self.ff(outputs)


@dataclass(eq=False)
class KvqAggregator(BatchAggregator):

    dim: int
    attn: nn.MultiheadAttention = field(init=False)
    num_heads: int = 4
    act_fn: Callable[[Tensor], Tensor] = field(init=False)
    blocks: nn.Sequential = field(init=False)
    num_blocks: int = 1
    dropout: float = 0.0
    hidden_dim: Optional[int] = None
    mean_query: bool = True

    def __post_init__(self) -> None:
        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(
                AttentionBlock(
                    dim=self.dim,
                    batch_size=self.batch_size,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
        if self.mean_query:
            agg_block = AttentionBlock(
                dim=self.dim,
                batch_size=self.batch_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                mean_query=True,
            )
        else:
            agg_block = BagMean(batch_size=self.batch_size)
        blocks.append(agg_block)
        self.blocks = nn.Sequential(*blocks)

    @implements(BatchAggregator)
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.blocks(inputs)


@dataclass(eq=False)
class GatedAggregator(BatchAggregator):

    dim: int
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
