from typing import Optional

from torch import nn
from torch.nn import functional as F

__all__ = ["BottleneckConvBlock"]


class BottleneckConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: int = 512,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        def _sub_block(
            _in_channels: int,
            _out_channels: int,
            _kernel_size: int,
            act: nn.Module = nn.ReLU(inplace=True),
        ) -> nn.Module:
            padding = int((((_kernel_size + 1) / 2) - 1))
            block = [nn.Conv2d(_in_channels, _out_channels, _kernel_size, 1, padding)]
            if use_bn:
                block.append(nn.BatchNorm2d(_out_channels))
            if act is not None:
                block.append(act)

            return block

        layers = [
            *_sub_block(in_channels, hidden_channels, 3),
            *_sub_block(hidden_channels, hidden_channels, 1),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ]

        for m in layers[:-1]:
            if hasattr(m, "weight"):
                nn.init.xavier_normal_(m.weight)
        # Initialize final kernel to zero so the coupling layer initially performs
        # an identity mapping
        nn.init.uniform_(layers[-1].weight, a=-1e-3, b=1e-3)

        self.sub_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.sub_blocks(x)
