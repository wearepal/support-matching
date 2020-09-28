from typing import List, Optional, Tuple

import torch.nn as nn

from shared.layers import View

__all__ = ["conv_autoencoder", "fc_autoencoder"]


def gated_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GLU(dim=1),
    )


def gated_up_conv(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.GLU(dim=1),
    )


def conv_autoencoder(
    input_shape,
    initial_hidden_channels: int,
    levels: int,
    encoding_dim,
    decoding_dim,
    variational: bool,
    decoder_out_act: Optional[nn.Module] = None,
) -> Tuple[nn.Sequential, nn.Sequential, Tuple[int, int, int]]:
    encoder: List[nn.Module] = []
    decoder: List[nn.Module] = []
    c_in, height, width = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        encoder.append(
            nn.Sequential(
                gated_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                gated_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
            )
        )

        decoder.append(
            nn.Sequential(
                # inverted order
                gated_up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0),
                gated_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
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

    enc_shape = (encoder_out_dim,)

    return encoder, decoder, enc_shape


def _linear_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def fc_autoencoder(
    input_shape: Tuple[int, ...],
    hidden_channels: int,
    levels: int,
    encoding_dim: int,
    variational: bool,
) -> Tuple[nn.Sequential, nn.Sequential, Tuple[int]]:
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

    if not variational:
        # whiten the encoding
        encoder += [nn.BatchNorm1d(encoder_out_dim, affine=False)]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoding_dim,)

    return encoder, decoder, enc_shape
