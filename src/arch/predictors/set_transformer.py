from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter

from .base import PredictorFactory, PredictorFactoryOut

__all__ = ["SetTransformer"]


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True))
        self.mh = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:  # type: ignore
        H = self.norm1(K + self.mh(key=K, query=Q, value=K, need_weights=False)[0])
        out = self.norm2(H + self.fc(H))
        return out


class SetAttentionBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int) -> None:
        super().__init__()
        self.mab = MultiheadAttentionBlock(embed_dim=dim_in)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.mab(x, x)


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int):
        super().__init__()
        self.inducing_points = Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab1 = MultiheadAttentionBlock(embed_dim=dim_out, num_heads=num_heads)
        self.mab2 = MultiheadAttentionBlock(embed_dim=dim_in, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        H = self.mab1(self.inducing_points.repeat(x.size(0), 1, 1), x)
        return self.mab2(x, H)


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int):
        super().__init__()
        self.seed_vectors = Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MultiheadAttentionBlock(embed_dim=dim, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.mab(self.seed_vectors.repeat(x.size(0), 1, 1), x)


class _SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        num_outputs: int,
        num_inds: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(hidden_dim, hidden_dim, num_heads, num_inds),
            InducedSetAttentionBlock(hidden_dim, hidden_dim, num_heads, num_inds),
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(hidden_dim, num_heads, num_outputs),
            SetAttentionBlock(hidden_dim, hidden_dim, num_heads),
            SetAttentionBlock(hidden_dim, hidden_dim, num_heads),
        )
        self.predictor = nn.Linear(hidden_dim * num_inds, target_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        out = self.embedder(x).unsqueeze(1)
        out = self.decoder(self.encoder(out))
        out = out.flatten(start_dim=1).sum(0)
        return self.predictor(out)


@dataclass
class SetTransformer(PredictorFactory):
    num_outputs: int
    num_inds: int = 32
    hidden_dim: int = 128
    num_heads: int = 4

    def __call__(
        self, input_dim: int, *, target_dim: int, **kwargs: Any
    ) -> PredictorFactoryOut[_SetTransformer]:
        st = _SetTransformer(
            in_dim=input_dim,
            target_dim=target_dim,
            num_outputs=self.num_outputs,
            num_inds=self.num_inds,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )
        return st, target_dim
