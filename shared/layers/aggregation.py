"""Modules that aggregate over a batch."""
from __future__ import annotations
from typing import Callable
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from shared.models.configs.classifiers import ModelFactory

__all__ = ["Aggregator", "KvqAttentionAggregator", "GatedAttentionAggregator"]


class Aggregator(nn.Module):
    output_dim: int
    bag_size: int

    def __init__(self, bag_size: int) -> None:
        super().__init__()
        self.bag_size = bag_size

    def bag_batch(self, batch: Tensor) -> Tensor:
        """
        Reshape a batch so that it's a batch of bags.

        This is the only certified way of producing bags. Use all other methods at your own risk.
        """
        return batch.view(-1, self.bag_size, *batch.shape[1:])


class KvqAttentionAggregator(Aggregator):

    act: Callable[[Tensor], Tensor]

    def __init__(
        self,
        latent_dim: int,
        *,
        bag_size: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
        final_proj: ModelFactory | None = None,
        output_dim: int = 1,
    ):
        super().__init__(bag_size=bag_size)
        self.latent_dim = latent_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=1, dropout=dropout, bias=True
        )
        if final_proj is not None:
            self.final_proj = final_proj(latent_dim, output_dim)
        else:
            self.final_proj = nn.Linear(latent_dim, output_dim)

        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation {activation}")
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        # `attn` expects the *second* dimension to be the batch dimension
        # that's why we have to transpose here
        inputs_batched = self.bag_batch(inputs).transpose(0, 1)  # shape: (bag, batch, latent)
        # for the query we just use an average of the bags
        query = inputs_batched.mean(dim=0, keepdim=True)
        key = inputs_batched
        value = key
        output = self.act(self.attn(query=query, key=key, value=value, need_weights=False)[0])
        return self.final_proj(output.view(-1, self.latent_dim))


class GatedAttentionAggregator(Aggregator):
    def __init__(
        self,
        in_dim: int,
        bag_size: int,
        embed_dim: int = 128,
        final_proj: ModelFactory | None = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__(bag_size=bag_size)
        self.V = nn.Parameter(torch.empty(embed_dim, in_dim))
        self.U = nn.Parameter(torch.empty(embed_dim, in_dim))
        self.w = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.w)
        if final_proj is not None:
            self.final_proj = final_proj(in_dim, output_dim)
        else:
            self.final_proj = nn.Linear(in_dim, output_dim)
        self.output_dim = output_dim
        self.attention_weights: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.tanh(inputs @ self.V.t()) * torch.sigmoid(inputs @ self.U.t()) @ self.w.t()
        logits_batched = self.bag_batch(logits)
        weights = logits_batched.softmax(dim=1)
        self.attention_weights = weights.squeeze(-1).detach().cpu()
        inputs_batched = self.bag_batch(inputs)
        weighted = torch.sum(weights * inputs_batched, dim=1, keepdim=False)
        return self.final_proj(weighted)
