from __future__ import annotations
from typing import Sequence

import torch.nn as nn

from shared.layers.misc import View

__all__ = ["conv_autoencoder", "fc_autoencoder"]


def _down_conv(in_channels: int, *, out_channels: int, kernel_size: int, stride: int, padding: int):
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
):
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


def conv_autoencoder(
    input_shape: Sequence[int],
    *,
    initial_hidden_channels: int,
    levels: int,
    encoding_dim: int,
    decoding_dim: int,
    variational: bool,
    decoder_out_act: nn.Module | None = None,
) -> tuple[nn.Sequential, nn.Sequential, int]:
    encoder_ls: list[nn.Module] = []
    decoder_ls: list[nn.Module] = []
    c_in, height, width = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
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
                    c_out, out_channels=c_out, kernel_size=4, stride=2, padding=1, output_padding=0
                ),
                _down_conv(c_out, out_channels=c_in, kernel_size=3, stride=1, padding=1),
            )
        )

        height //= 2
        width //= 2

    encoder_out_dim = 2 * encoding_dim if variational else encoding_dim

    flattened_size = c_out * height * width
    encoder_ls += [nn.Flatten()]
    encoder_ls += [nn.Linear(flattened_size, encoder_out_dim)]

    decoder_ls += [View((c_out, height, width))]
    decoder_ls += [nn.Linear(encoder_out_dim, flattened_size)]
    decoder_ls = decoder_ls[::-1]
    decoder_ls += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

    if decoder_out_act is not None:
        decoder_ls += [decoder_out_act]

    encoder = nn.Sequential(*encoder_ls)
    decoder = nn.Sequential(*decoder_ls)

    return encoder, decoder, encoder_out_dim


def _linear_block(in_channels: int, *, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def fc_autoencoder(
    input_shape: tuple[int, ...],
    *,
    hidden_channels: int,
    levels: int,
    encoding_dim: int,
    variational: bool,
) -> tuple[nn.Sequential, nn.Sequential, int]:
    encoder_ls = []
    decoder_ls = []

    c_in = input_shape[0]
    c_out = hidden_channels

    for _ in range(levels):
        encoder_ls += [_linear_block(c_in, out_channels=c_out)]
        decoder_ls += [_linear_block(c_out, out_channels=c_in)]
        c_in = c_out

    encoder_out_dim = 2 * encoding_dim if variational else encoding_dim

    encoder_ls += [_linear_block(c_in, out_channels=encoder_out_dim)]
    decoder_ls += [_linear_block(encoding_dim, out_channels=c_in)]
    decoder_ls = decoder_ls[::-1]

    # if not variational:
    #     # whiten the encoding
    #     encoder += [nn.BatchNorm1d(encoder_out_dim, affine=False)]

    encoder = nn.Sequential(*encoder_ls)
    decoder = nn.Sequential(*decoder_ls)

    return encoder, decoder, encoding_dim
