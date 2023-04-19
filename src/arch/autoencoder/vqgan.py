# Code derived from https://github.com/CompVis/taming-transformers/tree/master/taming/models
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from typing_extensions import override

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base import AeFactory, AePair

__all__ = ["VqGanAe"]


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, *, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):  # type: ignore
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, *, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        if self.conv is not None:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = w_.softmax(dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        in_channels: int,
        resolution: int,
        latent_dim: int,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        flattened_size = np.prod((block_in, curr_res, curr_res))
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, out_features=latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        # timestep embedding
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])  # type: ignore
                if len(self.down[i_level].attn) > 0:  # type: ignore
                    h = self.down[i_level].attn[i_block](h)  # type: ignore
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))  # type: ignore

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)  # type: ignore
        h = self.mid.attn_1(h)  # type: ignore
        h = self.mid.block_2(h)  # type: ignore

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        return self.to_latent(h)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_channels: int,
        attn_resolutions: Sequence[int],
        resolution: int,
        latent_dim: int,
        num_res_blocks: int,
        resamp_with_conv: bool = True,
        give_pre_end: bool = False,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        unflattened_size = (block_in, curr_res, curr_res)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, np.prod(unflattened_size)),
            nn.Unflatten(dim=1, unflattened_size=unflattened_size),
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:  # type: ignore
        # z to block_in
        h = self.from_latent(z)

        # middle
        h = self.mid.block_1(h)  # type: ignore
        h = self.mid.attn_1(h)  # type: ignore
        h = self.mid.block_2(h)  # type: ignore

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)  # type: ignore
                if len(self.up[i_level].attn) > 0:  # type: ignore
                    h = self.up[i_level].attn[i_block](h)  # type: ignore
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # type: ignore

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


@dataclass(eq=False)
class VqGanAe(AeFactory):
    attn_resolutions: List[int]
    ch_mult: List[int]
    latent_dim: int
    init_chans: int
    num_res_blocks: int
    dropout: float = 0.0

    @override
    def __call__(self, input_shape: Tuple[int, int, int]) -> AePair[Encoder, Decoder]:
        c, h, w = input_shape
        if h != w:
            raise ValueError(
                f"{self.__class__.__name__} expects input images to be square but received"
                f" input of resolution {h} X {w}."
            )

        encoder = Encoder(
            in_channels=c,
            resolution=h,
            ch=self.init_chans,
            ch_mult=self.ch_mult,
            latent_dim=self.latent_dim,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            resamp_with_conv=True,
        )
        decoder = Decoder(
            out_channels=c,
            resolution=h,
            ch=self.init_chans,
            ch_mult=self.ch_mult,
            latent_dim=self.latent_dim,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            resamp_with_conv=True,
        )
        return AePair(encoder=encoder, decoder=decoder, latent_dim=self.latent_dim)
