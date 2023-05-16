"""Modules that aggregate over a batch."""
from dataclasses import dataclass, field
from typing import Optional
from typing_extensions import override

from einops import rearrange
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


def bags_to_batch(batch: Tensor) -> Tensor:
    """
    Invert the ``batch_to_bags`` function.
    """
    return batch.movedim(0, -1).reshape(-1, *batch.shape[2:])


@dataclass(repr=False, eq=False)
class BatchAggregator(DcModule):
    batch_size: int

    def batch_to_bags(self, batch: Tensor) -> Tensor:
        """
        Reshape a batch so that it's a batch of bags.

        This is the only certified way of producing bags. Use all other methods at your own risk.
        """
        return batch_to_bags(batch=batch, batch_size=self.batch_size)


@dataclass(repr=False, eq=False)
class BagMean(BatchAggregator):
    @override
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

    @override
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
        self.head_dim = dim if head_dim is None else head_dim
        self._scale = self.head_dim**-0.5
        self.embed_dim = self.head_dim * self.num_heads
        self.to_qkv = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.embed_dim * 3, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        project_out = not (self.num_heads == 1 and self.head_dim == self.dim)
        self.post_attn = nn.Linear(self.embed_dim, self.dim) if project_out else nn.Identity()
        self.ffw = FeedForward(dim=self.dim, hidden_dim=self.dim)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        qkv = self.to_qkv(inputs)
        qkv = batch_to_bags(
            qkv, batch_size=self.batch_size
        )  # [batch, bag, 3 * head_dim * num_heads]
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv.chunk(3, dim=-1)
        )
        # qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        # q, k, v = qkv.chunk(3, dim=-1)
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
        outputs = bags_to_batch(outputs)
        # If not reducing (mean_query==False) then insert connections
        outputs = self.post_attn(outputs) + inputs
        return self.ffw(outputs) + outputs


@dataclass(repr=False, eq=False)
class KvqAggregator(BatchAggregator):
    dim: int
    num_blocks: int = 1
    dropout: float = 0.0
    mean_query: bool = True
    num_heads: int = 8
    head_dim: Optional[int] = 64

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
        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    @override
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        return self.blocks(inputs)


@dataclass(repr=False, eq=False)
class GatedAggregator(BatchAggregator):
    dim: int
    attention_weights: Optional[Tensor] = field(
        init=False, default=None, metadata={"omegaconf_ignore": True}
    )

    def __post_init__(self) -> None:
        self.v: Parameter = Parameter(torch.empty(self.dim, self.dim), requires_grad=True)
        self.u: Parameter = Parameter(torch.empty(self.dim, self.dim), requires_grad=True)
        self.w: Parameter = Parameter(torch.empty(1, self.dim), requires_grad=True)
        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.w)

    @override
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        inputs = inputs.flatten(start_dim=1)
        logits = torch.tanh(inputs @ self.v.t()) * torch.sigmoid(inputs @ self.u.t()) @ self.w.t()
        logits_batched = self.batch_to_bags(logits)
        weights = logits_batched.softmax(dim=1)
        self.attention_weights = weights.squeeze(-1).detach().cpu()
        inputs_batched = self.batch_to_bags(inputs)
        return torch.sum(weights * inputs_batched, dim=1, keepdim=False)
