from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor

from shared.configs import MMDKer

__all__ = ["mmd2"]


def _dot_kernel(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    return xx_gm, xy_gm, yy_gm, 0.0


def _mix_rq_kernel(
    x: Tensor,
    y: Tensor,
    scales: Optional[Sequence[float]] = None,
    wts: Optional[Sequence[float]] = None,
    add_dot: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Rational quadratic kernel
    http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    """
    scales = (0.1, 1.0, 10.0) or scales
    wts = [1.0] * len(scales) or wts

    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 0)

    def pad_second(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 1)

    xx_sqnorm = torch.clamp(-2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms), min=0.0)
    xy_sqnorm = torch.clamp(-2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms), min=0.0)
    yy_sqnorm = torch.clamp(-2 * yy_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms), min=0.0)

    k_xx, k_xy, k_yy = (
        x.new_zeros(xx_sqnorm.shape),
        x.new_zeros(xy_sqnorm.shape),
        x.new_zeros(yy_sqnorm.shape),
    )

    for alpha, wt in zip(scales, wts):
        log_xx = torch.log(1.0 + xx_sqnorm / (2.0 * alpha))
        k_xx += wt * torch.exp(-alpha * log_xx)
        log_xy = torch.log(1.0 + xy_sqnorm / (2.0 * alpha))
        k_xy += wt * torch.exp(-alpha * log_xy)
        log_yy = torch.log(1.0 + yy_sqnorm / (2.0 * alpha))
        k_yy += wt * torch.exp(-alpha * log_yy)

    if add_dot > 0:
        k_xy += add_dot * xy_gm
        k_xx += add_dot * xx_gm
        k_yy += add_dot * yy_gm

    return k_xx, k_xy, k_yy, sum(wts)


def _mix_rbf_kernel(
    x: Tensor,
    y: Tensor,
    scales: Optional[Sequence[float]] = None,
    wts: Optional[Sequence[float]] = None,
    add_dot: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """"""
    scales = scales or (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    wts = wts or ([1.0] * len(scales))

    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 0)

    def pad_second(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 1)

    xx_sqnorm = -2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms)
    xy_sqnorm = -2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms)
    yy_sqnorm = -2 * yy_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms)

    k_xx, k_xy, k_yy = (
        x.new_zeros(xx_sqnorm.shape),
        x.new_zeros(xy_sqnorm.shape),
        x.new_zeros(yy_sqnorm.shape),
    )

    for sigma, wt in zip(scales, wts):
        gamma = 1.0 / (2 * sigma ** 2)
        k_xx = wt * torch.exp(-gamma * xx_sqnorm)
        k_xy += wt * torch.exp(-gamma * xy_sqnorm)
        k_yy += wt * torch.exp(-gamma * yy_sqnorm)

    return k_xx, k_xy, k_yy, sum(wts)


def _mmd2(
    k_xx: Tensor, k_xy: Tensor, k_yy: Tensor, const_diagonal: float = 0.0, biased: bool = False
) -> Tensor:
    m = k_xx.size(0)
    n = k_yy.size(0)

    if biased:
        mmd2 = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    else:
        if const_diagonal is not False:
            trace_x = torch.tensor(m)
            trace_y = torch.tensor(n)
        else:
            trace_x = k_xx.trace()
            trace_y = k_yy.trace()
        mmd2 = (
            (k_xx.sum() - trace_x) / (m * (m - 1))
            + (k_yy.sum() - trace_y) / (n * (n - 1))
            - (2 * k_xy.sum() / (m * n))
        )

    return mmd2


def mmd2(
    x: Tensor,
    y: Tensor,
    kernel: MMDKer = MMDKer.rq,
    biased: bool = False,
    **kwargs: Any,
) -> Tensor:
    if kernel == MMDKer.linear:
        kernel_out = _dot_kernel(x, y)
    elif kernel == MMDKer.rbf:
        kernel_out = _mix_rbf_kernel(x, y, **kwargs)
    else:
        kernel_out = _mix_rq_kernel(x, y, **kwargs)
    return _mmd2(*kernel_out, biased)
