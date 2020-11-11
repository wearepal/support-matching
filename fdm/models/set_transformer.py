import torch
import torch.nn as nn
from torch.tensor import Tensor


__all__ = ["SetTransformer", "MAB", "SAB", "ISAB", "PMA"]


class MAB(nn.Module):
    def __init__(self, embed_dim: int, kdim: int, vdim: int, num_heads: int) -> None:
        super().__init__()
        self.ln0 = nn.LayerNorm(vdim)
        self.ln1 = nn.LayerNorm(vdim)
        self.fc = nn.Sequential(nn.Linear(vdim, vdim), nn.ReLU(inplace=True))
        self.mh = nn.MultiheadAttention(
            embed_dim=embed_dim, kdim=kdim, vdim=vdim, num_heads=num_heads
        )

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        H = self.ln0(K + self.mh(key=K, query=Q, value=K))
        return self.ln1(H + self.fc(H))


# class MAB(nn.Module):
#     def __init__(self, embed_dim: int, kdim: int, vdim: int, num_heads: int):
#         super().__init__()
#         self.vdim = vdim
#         self.num_heads = num_heads
#         self.fc_q = nn.Linear(embed_dim, vdim)
#         self.fc_k = nn.Linear(kdim, vdim)
#         self.fc_v = nn.Linear(kdim, vdim)
#         self.ln0 = nn.LayerNorm(vdim)
#         self.ln1 = nn.LayerNorm(vdim)
#         self.fc_o = nn.Linear(vdim, vdim)

#     def forward(self, Q, K):
#         Q = self.fc_q(Q)
#         K, V = self.fc_k(K), self.fc_v(K)

#         dim_split = self.vdim // self.num_heads
#         Q_ = torch.cat(Q.split(dim_split, 2), 0)
#         K_ = torch.cat(K.split(dim_split, 2), 0)
#         V_ = torch.cat(V.split(dim_split, 2), 0)

#         A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.vdim), 2)
#         O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
#         O = self.ln0(O)
#         O = O + self.fc_o(O).relu()
#         O = self.ln1(O)
#         return O


class SAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int) -> None:
        super().__init__()
        self.mab = MAB(embed_dim=dim_in, kdim=dim_in, vdim=dim_out, num_heads=num_heads)

    def forward(self, X: Tensor) -> Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab0 = MAB(embed_dim=dim_out, kdim=dim_in, vdim=dim_out, num_heads=num_heads)
        self.mab1 = MAB(embed_dim=dim_in, kdim=dim_out, vdim=dim_out, num_heads=num_heads)

    def forward(self, X):
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embed_dim=dim, kdim=dim, vdim=dim, num_heads=num_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


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
            ISAB(in_dim, hidden_dim, num_heads, num_inds),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds),
        )
        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, num_outputs),
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, X):
        return self.dec(self.enc(X))
