from __future__ import annotations
from typing import List, Optional, Tuple

import torch.nn as nn

from shared.layers import View
from shared.layers.misc import UnitNormLayer

__all__ = ["conv_autoencoder", "fc_autoencoder"]


def down_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.SiLU(),
    )


def up_conv(in_channels, out_channels, kernel_size, stride, padding, output_padding):
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
    input_shape,
    initial_hidden_channels: int,
    levels: int,
    encoding_dim,
    decoding_dim,
    variational: bool,
    decoder_out_act: Optional[nn.Module] = None,
) -> Tuple[nn.Sequential, nn.Sequential, int]:
    encoder: list[nn.Module] = []
    decoder: list[nn.Module] = []
    c_in, height, width = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        encoder.append(
            nn.Sequential(
                down_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                down_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
            )
        )

        decoder.append(
            nn.Sequential(
                # inverted order
                up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0),
                down_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
            )
        )

        height //= 2
        width //= 2

    encoder_out_dim = 2 * encoding_dim if variational else encoding_dim

    flattened_size = c_out * height * width
    encoder += [nn.Flatten()]
    encoder += [nn.Linear(flattened_size, encoder_out_dim)]

    decoder += [View((c_out, height, width))]
    decoder += [nn.Linear(encoder_out_dim, flattened_size)]
    decoder = decoder[::-1]
    decoder += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

    if decoder_out_act is not None:
        decoder += [decoder_out_act]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    return encoder, decoder, encoder_out_dim


def _linear_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def fc_autoencoder(
    input_shape: Tuple[int, ...],
    hidden_channels: int,
    levels: int,
    encoding_dim: int,
    variational: bool,
) -> Tuple[nn.Sequential, nn.Sequential, int]:
    encoder = []
    decoder = []

    c_in = input_shape[0]
    c_out = hidden_channels

    for _ in range(levels):
        encoder += [_linear_block(c_in, c_out)]
        decoder += [_linear_block(c_out, c_in)]
        c_in = c_out

    encoder_out_dim = 2 * encoding_dim if variational else encoding_dim

    encoder += [_linear_block(c_in, encoder_out_dim)]
    decoder += [_linear_block(encoding_dim, c_in)]
    decoder = decoder[::-1]

    # if not variational:
    #     # whiten the encoding
    #     encoder += [nn.BatchNorm1d(encoder_out_dim, affine=False)]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    return encoder, decoder, encoding_dim
