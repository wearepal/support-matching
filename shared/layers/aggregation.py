"""Modules that aggregate over a batch."""
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import Literal

from shared.utils import ModelFn

__all__ = [
    "Aggregator",
    "KvqAttentionAggregator",
    "GatedAttentionAggregator",
]


class Aggregator(nn.Module):
    output_dim: int
    bag_size: int

    def __init__(self, bag_size: int = 1) -> None:
        super().__init__()
        self.bag_size = bag_size


class KvqAttentionAggregator(Aggregator):

    act: Callable[[Tensor], Tensor]

    def __init__(
        self,
        latent_dim: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
        final_proj: Optional[ModelFn] = None,
        output_dim: int = 1,
        bag_size: int = 1,
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
        # for the query we just use an average of all inputs
        inputs_batched = inputs.view(self.bag_size, -1, *inputs.shape[1:])
        query = inputs_batched.mean(dim=1, keepdim=True)
        # the second dimension is supposed to be the "batch size",
        # but we're aggregating over the batch, so we set this just to 1
        key = inputs.view(self.bag_size, -1, self.latent_dim)
        value = key
        output = self.act(self.attn(query=query, key=key, value=value, need_weights=False)[0])
        return self.final_proj(output.view(-1, self.latent_dim))


class GatedAttentionAggregator(Aggregator):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        final_proj: Optional[ModelFn] = None,
        output_dim: int = 1,
        bag_size: int = 1,
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

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.tanh(inputs @ self.V.t()) * torch.sigmoid(inputs @ self.U.t()) @ self.w.t()
        logits_batched = logits.view(-1, self.bag_size, 1)
        weights = logits_batched.softmax(dim=1)
        inputs_batched = inputs.view(-1, self.bag_size, *inputs.shape[1:])
        weighted = torch.sum(weights * inputs_batched, dim=1, keepdim=False)
        return self.final_proj(weighted)
