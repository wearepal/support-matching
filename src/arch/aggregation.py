"""Modules that aggregate over a batch."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from einops import rearrange
from ranzen import implements
from ranzen.torch import DcModule
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

__all__ = ["BatchAggregator", "GatedAggregator", "KvqAggregator", "BagMean"]


def batch_to_bags(batch: Tensor, *, batch_size: int) -> Tensor:
    """
    Reshape a batch so that it's a batch of bags.

    This is the only certified way of producing bags. Use all other methods at your own risk.
    """
    return batch.reshape(-1, *batch.shape[1:], batch_size).movedim(-1, 0)


def bags_to_batch(batch: Tensor, *, batch_size: int) -> Tensor:
    """
    Invert the ``batch_to_bags`` function.
    """
    return batch.movedim(0, -1).reshape(-1, *batch.shape[2:])


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
    def __init__(self, dim: int, *, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
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
        *,
        batch_size: int,
        num_heads: int = 8,
        head_dim: Optional[int] = 64,
        dropout: float = 0.0,
        mean_query: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.mean_query = mean_query
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim if head_dim is None else dim
        self.embed_dim = self.head_dim * num_heads
        self.to_qkv = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(dim, self.embed_dim * 3, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        project_out = not (num_heads == 1 and head_dim == dim)
        self.post_attn = nn.Linear(self.embed_dim, dim) if project_out else nn.Identity()
        self.ffw = FeedForward(dim=self.dim, hidden_dim=self.dim)
        self._scale = self.head_dim**-0.5

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        qkv = self.to_qkv(inputs)
        qkv = batch_to_bags(qkv, batch_size=self.batch_size)
        # shape: (batch, num_heads, bag, latent)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.mean_query:
            q = q.mean(2, keepdim=True)
        dots = q @ k.transpose(-1, -2) * self._scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        outputs = attn @ v
        outputs = rearrange(outputs, "b h n d -> b n (h d)", h=self.num_heads)

        if self.mean_query:
            outputs = self.post_attn(outputs.squeeze(1))
            return self.ffw(outputs)
        outputs = bags_to_batch(outputs, batch_size=self.batch_size)
        # If not reducing (mean_query==False) then insert a residual connection
        outputs = self.post_attn(outputs) + inputs
        return self.ffw(outputs) + outputs


@dataclass(eq=False)
class KvqAggregator(BatchAggregator):

    dim: int
    num_blocks: int = 1
    dropout: float = 0.0
    mean_query: bool = True
    num_heads: int = 8
    head_dim: Optional[int] = 64
    blocks: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(
                AttentionBlock(
                    dim=self.dim,
                    batch_size=self.batch_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout=self.dropout,
                    mean_query=False,
                )
            )
        if self.mean_query:
            agg_block = AttentionBlock(
                dim=self.dim,
                batch_size=self.batch_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
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
