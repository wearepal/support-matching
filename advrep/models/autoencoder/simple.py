from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from advrep.models.autoencoder.base import AutoEncoder, ReconstructionLoss, ZsTransform
from shared.layers.misc import View

__all__ = ["SimpleConvAE"]


class SimpleConvAE(AutoEncoder):
    def __init__(
        self,
        input_shape: Optional[Tuple[int]],
        *,
        latent_dim: int,
        zs_dim: int,
        zs_transform: ZsTransform = ZsTransform.none,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        recon_loss_fn: ReconstructionLoss = ReconstructionLoss.l2,
        levels: int = 4,
        init_chans: int = 32,
    ) -> None:
        self.levels = levels
        self.init_chans = init_chans
        super().__init__(
            input_shape=input_shape,
            zs_dim=zs_dim,
            latent_dim=latent_dim,
            zs_transform=zs_transform,
            feature_group_slices=feature_group_slices,
            lr=lr,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            recon_loss_fn=recon_loss_fn,
        )

    @staticmethod
    def _down_conv(
        in_channels: int, *, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
        )

    @staticmethod
    def _up_conv(
        in_channels: int,
        *,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
        )

    def _build(self, input_shape: Sequence[int]) -> tuple[nn.Sequential, nn.Sequential]:
        encoder_ls: list[nn.Module] = []
        decoder_ls: list[nn.Module] = []
        c_in, height, width = input_shape
        c_out = self.init_chans

        for level in range(self.levels):
            if level != 0:
                c_in = c_out
                c_out *= 2

            encoder_ls.append(
                nn.Sequential(
                    self._down_conv(c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
                    self._down_conv(c_out, out_channels=c_out, kernel_size=4, stride=2, padding=1),
                )
            )

            decoder_ls.append(
                nn.Sequential(
                    # inverted order
                    self._up_conv(
                        c_out,
                        out_channels=c_out,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                    ),
                    self._down_conv(c_out, out_channels=c_in, kernel_size=3, stride=1, padding=1),
                )
            )

            height //= 2
            width //= 2

        flattened_size = c_out * height * width
        encoder_ls += [nn.Flatten()]
        encoder_ls += [nn.Linear(flattened_size, self.latent_dim)]

        decoder_ls += [View((c_out, height, width))]
        decoder_ls += [nn.Linear(self.latent_dim, flattened_size)]
        decoder_ls = decoder_ls[::-1]
        decoder_ls += [
            nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, stride=1, padding=0)
        ]

        encoder = nn.Sequential(*encoder_ls)
        decoder = nn.Sequential(*decoder_ls)

        return encoder, decoder
