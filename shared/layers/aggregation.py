"""Modules that aggregate over a batch."""
from __future__ import annotations
from typing import Any
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from shared.models.factory import ModelFactory

__all__ = [
    "Aggregator",
    "GatedAttentionAggregator",
    "KvqAttentionAggregator",
    "ModelAggregatorWrapper",
]


class Aggregator(nn.Module):
    bag_size: int

    def __init__(self, bag_size: int, *, output_dim: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.bag_size = bag_size

    def batch_to_bags(self, batch: Tensor) -> Tensor:
        """
        Reshape a batch so that it's a batch of bags.

        This is the only certified way of producing bags. Use all other methods at your own risk.
        """
        return batch.view(self.bag_size, *batch.shape[1:], -1).movedim(-1, 0)


class KvqAttentionAggregator(Aggregator):
    def __init__(
        self,
        embed_dim: int,
        *,
        bag_size: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
        final_proj: ModelFactory | None = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__(bag_size=bag_size, output_dim=output_dim)
        self.latent_dim = embed_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=dropout, bias=True
        )
        if final_proj is not None:
            self.final_proj = final_proj(embed_dim, target_dim=output_dim)
        else:
            self.final_proj = nn.Linear(in_features=embed_dim, out_features=output_dim)

        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        # `attn` expects the *second* dimension to be the batch dimension
        # that's why we have to transpose here
        inputs_batched = self.batch_to_bags(inputs).transpose(0, 1)  # shape: (bag, batch, latent)
        # for the query we just use an average of the bags
        query = inputs_batched.mean(dim=0, keepdim=True)
        key = inputs_batched
        value = key
        output = self.act(self.attn(query=query, key=key, value=value, need_weights=False)[0])
        return self.final_proj(output.view(-1, self.latent_dim))


class GatedAttentionAggregator(Aggregator):
    def __init__(
        self,
        embed_dim: int,
        *,
        bag_size: int,
        final_proj: ModelFactory | None = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__(bag_size=bag_size, output_dim=output_dim)
        self.V = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.U = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.w = nn.Parameter(torch.empty(1, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.w)
        if final_proj is not None:
            self.final_proj = final_proj(embed_dim, target_dim=output_dim)
        else:
            self.final_proj = nn.Linear(in_features=embed_dim, out_features=output_dim)
        self.attention_weights: Tensor

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        inputs = inputs.flatten(start_dim=1)
        logits = torch.tanh(inputs @ self.V.t()) * torch.sigmoid(inputs @ self.U.t()) @ self.w.t()
        logits_batched = self.batch_to_bags(logits)
        weights = logits_batched.softmax(dim=1)
        self.attention_weights = weights.squeeze(-1).detach().cpu()
        inputs_batched = self.batch_to_bags(inputs)
        weighted = torch.sum(weights * inputs_batched, dim=1, keepdim=False)
        return self.final_proj(weighted)


class ModelAggregatorWrapper(ModelFactory[nn.Sequential]):
    def __init__(self, model_fn: ModelFactory, *, aggregator: Aggregator, input_dim: int) -> None:
        self.model_fn = model_fn
        self.aggregator = aggregator
        self.input_dim = input_dim

    def __call__(self, input_dim: int, *, target_dim: int) -> nn.Sequential:
        assert target_dim == self.aggregator.output_dim

        return nn.Sequential(
            self.model_fn(input_dim, target_dim=self.input_dim), nn.GELU(), self.aggregator
        )
