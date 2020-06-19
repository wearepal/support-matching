from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import Bijector


class SqueezeLayer(Bijector):
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor: int = downscale_factor

    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        squeeze_x = squeeze(x, self.downscale_factor)
        if sum_ldj is None:
            return squeeze_x, None
        else:
            return squeeze_x, sum_ldj

    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if sum_ldj is None:
            return unsqueeze_y, None
        else:
            return unsqueeze_y, sum_ldj


class UnsqueezeLayer(SqueezeLayer):
    def __init__(self, upscale_factor: int) -> None:
        super().__init__(downscale_factor=upscale_factor)

    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._inverse(x, sum_ldj)

    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._forward(y, sum_ldj)


def unsqueeze(input: Tensor, upscale_factor: int = 2) -> Tensor:
    """
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = int(in_channels // (upscale_factor ** 2))

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width
    )

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input: Tensor, downscale_factor: int = 2) -> Tensor:
    """
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = int(in_channels * (downscale_factor ** 2))

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


class HaarDownsampling(Bijector):
    """Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height."""

    def __init__(
        self, in_channels: Tensor, order_by_wavelet: bool = False, rebalance: float = 1.0
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.permute = order_by_wavelet
        self.last_jac = None

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i + 4 * j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def logdetjac(self, x: Tensor, reverse: bool) -> Tensor:
        fac = self.fac_rev if reverse else self.fac_fwd
        return x[0].nelement() / 4 * (np.log(16.0) + 4 * np.log(fac))

    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels)

        if self.permute:
            out = out[:, self.perm]
        out *= self.fac_fwd

        if sum_ldj is not None:
            sum_ldj -= self.logdetjac(x, False)
        return out, sum_ldj

    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if self.permute:
            x_perm = y[:, self.perm_inv]
        else:
            x_perm = y

        out = F.conv_transpose2d(
            x_perm * self.fac_rev, self.haar_weights, bias=None, stride=2, groups=self.in_channels
        )

        if sum_ldj is not None:
            sum_ldj += self.logdetjac(y, True)
        return out, sum_ldj
