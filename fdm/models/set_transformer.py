import torch
import torch.nn as nn
from torch.tensor import Tensor


__all__ = [
    "SetTransformer",
    "MultiheadAttentionBlock",
    "SetAttentionBlock",
    "InducedSetAttentionBlock",
    "PoolingMultiheadAttention",
]


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True))
        self.mh = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        Q_tiled = Q.view(-1, 1, Q.size(1))
        K_tiled = K.view(-1, 1, K.size(1))
        import pdb

        pdb.set_trace()
        H = self.norm1(K_tiled + self.mh(key=K_tiled, query=Q_tiled, value=K_tiled))
        out = self.norm2(H + self.fc(H))
        return out.view(-1, out.size(-1))


class SetAttentionBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int) -> None:
        super().__init__()
        self.mab = MultiheadAttentionBlock(embed_dim=dim_in)

    def forward(self, X: Tensor) -> Tensor:
        return self.mab(X, X)


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab1 = MultiheadAttentionBlock(embed_dim=dim_out, num_heads=num_heads)
        self.mab2 = MultiheadAttentionBlock(embed_dim=dim_in, num_heads=num_heads)

    def forward(self, X):
        H = self.mab1(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MultiheadAttentionBlock(embed_dim=dim, num_heads=num_heads)

    def forward(self, X):
        return self.mab(self.seed_vectors.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        num_outputs: int,
        num_inds: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.enc = nn.Sequential(
            InducedSetAttentionBlock(in_dim, hidden_dim, num_heads, num_inds),
            InducedSetAttentionBlock(hidden_dim, hidden_dim, num_heads, num_inds),
        )
        self.dec = nn.Sequential(
            PoolingMultiheadAttention(hidden_dim, num_heads, num_outputs),
            SetAttentionBlock(hidden_dim, hidden_dim, num_heads),
            SetAttentionBlock(hidden_dim, hidden_dim, num_heads),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, X):
        return self.dec(self.enc(X))
