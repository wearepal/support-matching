from __future__ import annotations
from dataclasses import dataclass

from ranzen import implements
import torch.nn as nn

from .base import AeFactory, AePair

__all__ = ["SimpleConvAE"]


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


@dataclass(eq=False)
class SimpleConvAE(AeFactory):
    latent_dim: int = 128
    levels: int = 4
    init_chans: int = 32

    @implements(AeFactory)
    def __call__(self, input_shape: tuple[int, int, int]) -> AePair[nn.Sequential, nn.Sequential]:
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
                    _down_conv(c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
                    _down_conv(c_out, out_channels=c_out, kernel_size=4, stride=2, padding=1),
                )
            )

            decoder_ls.append(
                nn.Sequential(
                    # inverted order
                    _up_conv(
                        c_out,
                        out_channels=c_out,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                    ),
                    _down_conv(c_out, out_channels=c_in, kernel_size=3, stride=1, padding=1),
                )
            )

            height //= 2
            width //= 2

        flattened_size = c_out * height * width
        encoder_ls += [nn.Flatten()]
        encoder_ls += [nn.Linear(flattened_size, self.latent_dim)]

        decoder_ls += [nn.Unflatten(dim=1, unflattened_size=(c_out, height, width))]
        decoder_ls += [nn.Linear(self.latent_dim, flattened_size)]
        decoder_ls = decoder_ls[::-1]
        decoder_ls += [
            nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, stride=1, padding=0)
        ]

        encoder = nn.Sequential(*encoder_ls)
        decoder = nn.Sequential(*decoder_ls)

        return AePair(
            encoder=encoder,
            decoder=decoder,
            latent_dim=self.latent_dim,
        )
