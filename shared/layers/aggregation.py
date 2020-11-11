"""Modules that aggregate over a batch."""
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.nn as nn
from typing_extensions import Literal

from shared.utils import ModelFn

__all__ = [
    "Aggregator",
    "AttentionAggregator",
    "SimpleAggregator",
    "SimpleAggregatorT",
    "GatedAttention",
]


class Aggregator(nn.Module):
    output_dim: int


class AttentionAggregator(Aggregator):
    def __init__(
        self,
        latent_dim: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
        final_proj: Optional[ModelFn] = None,
        output_dim: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=1, dropout=dropout, bias=True
        )
        if final_proj is not None:
            self.final_proj = final_proj(latent_dim, output_dim)
        else:
            self.final_proj = nn.Linear(latent_dim, output_dim)
        self.act: Callable[[Tensor], Tensor]
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation {activation}")
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        # for the query we just use an average of all inputs
        query = inputs.mean(dim=0).view(1, 1, self.latent_dim)
        # the second dimension is supposed to be the "batch size",
        # but we're aggregating over the batch, so we set this just to 1
        key = inputs.view(-1, 1, self.latent_dim)
        value = key
        output = self.act(self.attn(query=query, key=key, value=value, need_weights=False)[0])
        return self.final_proj(output.view(1, self.latent_dim)).view(1, -1)


class SimpleAggregator(Aggregator):
    def __init__(
        self, *, latent_dim: int, final_proj: Optional[ModelFn] = None, output_dim: int = 1
    ):
        super().__init__()
        self.weight_proj = nn.Linear(latent_dim, 1)
        if final_proj is not None:
            self.final_proj = final_proj(latent_dim, output_dim)
        else:
            self.final_proj = nn.Linear(latent_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        weights = F.softmax(self.weight_proj(inputs), dim=0)
        weighted = torch.sum(weights * inputs, dim=0, keepdim=True)
        return self.final_proj(weighted)


class SimpleAggregatorT(Aggregator):
    """Transposed version of `SimpleAggregator`."""

    def __init__(
        self, *, batch_dim: int, final_proj: Optional[ModelFn] = None, output_dim: int = 1
    ):
        super().__init__()
        self.weight_proj = nn.Linear(batch_dim, 1)
        if final_proj is not None:
            self.final_proj = final_proj(batch_dim, output_dim)
        else:
            self.final_proj = nn.Linear(batch_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        weights = F.softmax(self.weight_proj(inputs.t()).view(-1), dim=-1)
        weighted = torch.sum(weights * inputs, dim=-1)
        return self.final_proj(weighted.view(1, -1))


class GatedAttention(Aggregator):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 128,
        final_proj: Optional[ModelFn] = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.V = nn.Parameter(torch.empty(latent_dim, in_dim))
        self.U = nn.Parameter(torch.empty(latent_dim, in_dim))
        self.w = nn.Parameter(torch.empty(1, latent_dim))
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.w)
        if final_proj is not None:
            self.final_proj = final_proj(latent_dim, output_dim)
        else:
            self.final_proj = nn.Linear(latent_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.tanh(inputs @ self.V.t()) * torch.sigmoid(inputs @ self.U.t()) @ self.w.t()
        weights = logits.softmax(dim=0)
        weighted = torch.sum(weights * inputs, dim=0, keepdim=True)
        return self.final_proj(weighted)
