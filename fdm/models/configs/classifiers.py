from typing import Protocol, Union, Sequence, Optional, List

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from fdm.models.resnet import ResidualNet
from shared.utils import prod

__all__ = [
    "linear_resnet",
    "mp_28x28_net",
    "mp_32x32_net",
    "mp_64x64_net",
    "residual_64x64_net",
    "strided_28x28_net",
    "fc_net",
    "ModelFn",
]


class ModelFn(Protocol):
    def __call__(
        self, input_dim: int, target_dim: int, **model_kwargs: Union[float, str, bool]
    ) -> nn.Module:
        ...


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super(ResidualBlock, self).__init__()
        block: List[nn.Module] = []

        if batch_norm:
            block += [nn.BatchNorm2d(in_channels)]
        block += [nn.LeakyReLU(inplace=True)]
        block += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        if batch_norm:
            block += [nn.BatchNorm2d(out_channels)]
        block += [nn.LeakyReLU(inplace=True)]
        block += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
        self.block = nn.Sequential(*block)

        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.block(x)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return out


def linear_resnet(in_dim, target_dim, hidden_channels=512, num_blocks=4, batch_norm=False):

    act = F.relu if batch_norm else F.selu
    layers = [
        nn.Flatten(),
        ResidualNet(
            in_features=in_dim,
            out_features=target_dim,
            hidden_features=hidden_channels,
            num_blocks=num_blocks,
            activation=act,
            dropout_probability=0.0,
            use_batch_norm=batch_norm,
        ),
    ]
    return nn.Sequential(*layers)


def mp_64x64_net(input_dim, target_dim, batch_norm=True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 5, 1, 0))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def resnet_50_ft(input_dim, target_dim, freeze=True, contexted=True):
    net = resnet50(contexted=contexted)
    # net = resnet18(contexted=contexted)
    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    # net.fc = nn.Linear(512, target_dim)
    net.fc = nn.Linear(2048, target_dim)

    return net


def mp_32x32_net(input_dim: int, target_dim: int, batch_norm: bool = True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 5, 1, 0))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def mp_28x28_net(input_dim, target_dim, batch_norm=True):
    def conv_block(in_dim, out_dim, kernel_size, stride):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1)]
        if batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def strided_28x28_net(input_dim, target_dim, batch_norm=True):
    def conv_block(in_dim, out_dim, kernel_size, stride):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1)]
        if batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 3, 1))
    layers.extend(conv_block(64, 64, 4, 2))

    layers.extend(conv_block(64, 128, 3, 1))
    layers.extend(conv_block(128, 128, 4, 2))

    layers.extend(conv_block(128, 256, 3, 1))
    layers.extend(conv_block(256, 256, 4, 2))

    layers.extend(conv_block(256, 512, 3, 1))
    layers.extend(conv_block(512, 512, 4, 2))

    layers += [nn.AdaptiveAvgPool2d(1)]
    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def residual_64x64_net(input_dim, target_dim, batch_norm=False):

    layers = []
    for out_channels in [64, 128, 256, 512]:
        layers += [
            ResidualBlock(
                in_channels=input_dim, out_channels=out_channels, stride=2, batch_norm=batch_norm
            )
        ]
        input_dim = out_channels

    layers.append(
        ResidualBlock(in_channels=512, out_channels=1024, stride=1, batch_norm=batch_norm)
    )

    layers += [nn.AdaptiveAvgPool2d(1)]
    layers += [nn.Flatten()]
    layers += [nn.Linear(1024, target_dim)]

    return nn.Sequential(*layers)


def fc_net(
    input_dim: Union[int, Sequence[int]], target_dim, hidden_dims: Optional[Sequence[int]] = None
):
    hidden_dims = hidden_dims or []

    def fc_block(in_dim, out_dim):
        _block = []
        _block += [nn.Linear(in_dim, out_dim)]
        _block += [nn.SELU()]
        return _block

    layers = [nn.Flatten()]
    if not isinstance(input_dim, int):
        input_dim = prod(input_dim)

    for output_dim in hidden_dims:
        layers.extend(fc_block(input_dim, output_dim))
        input_dim = output_dim

    layers.append(nn.Linear(input_dim, target_dim))

    return nn.Sequential(*layers)
