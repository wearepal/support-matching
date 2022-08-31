from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from ranzen import implements
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import AeFactory, AeFactoryOut

__all__ = ["ResNetAE"]


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(
        self, size: int | None = None, scale_factor: float | list[float] | None = None
    ) -> None:
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes: int, *, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, *, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(
    in_planes: int, *, out_planes: int, scale: float = 1
) -> nn.Sequential | nn.Conv2d:
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes=out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes=out_planes))


def resize_conv1x1(
    in_planes: int, *, out_planes: int, scale: float = 1
) -> nn.Sequential | nn.Conv2d:
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes=in_planes, out_planes=out_planes)
    return nn.Sequential(
        Interpolate(scale_factor=scale), conv1x1(in_planes=in_planes, out_planes=out_planes)
    )


class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EncoderBottleneck(nn.Module):
    """ResNet bottleneck, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L75."""

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = conv1x1(in_planes=inplanes, out_planes=width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(in_planes=width, out_planes=width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(in_planes=width, out_planes=planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, scale: float = 1, upsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = resize_conv3x3(in_planes=inplanes, out_planes=inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, out_planes=planes, scale=scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    """ResNet bottleneck, but convs replaced with resize convs."""

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, scale: float = 1, upsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(in_planes=inplanes, out_planes=width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(in_planes=width, out_planes=width, scale=scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(in_planes=width, out_planes=planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        block: type[EncoderBlock] | type[EncoderBottleneck],
        layers: list[int],
        first_conv: bool = False,
        maxpool1: bool = False,
        latent_dim: int = 512,
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(in_features=512, out_features=latent_dim)

    def _make_layer(
        self,
        block: type[EncoderBlock] | type[EncoderBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    in_planes=self.inplanes, out_planes=planes * block.expansion, stride=stride
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projector(x)


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(
        self,
        block: type[DecoderBlock] | type[DecoderBottleneck],
        layers: list[int],
        latent_dim: int,
        input_height: int,
        first_conv: bool = False,
        maxpool1: bool = False,
    ) -> None:
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(
            64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(
        self,
        block: type[DecoderBlock] | type[DecoderBottleneck],
        planes: int,
        blocks: int,
        scale: float = 1,
    ) -> nn.Sequential:
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(
                    in_planes=self.inplanes, out_planes=planes * block.expansion, scale=scale
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


def resnet18_encoder(first_conv: bool, *, maxpool1: bool, latent_dim: int):
    return ResNetEncoder(
        EncoderBlock,
        layers=[2, 2, 2, 2],
        first_conv=first_conv,
        maxpool1=maxpool1,
        latent_dim=latent_dim,
    )


def resnet18_decoder(latent_dim: int, *, input_height: int, first_conv: bool, maxpool1: bool):
    return ResNetDecoder(
        DecoderBlock,
        layers=[2, 2, 2, 2],
        latent_dim=latent_dim,
        input_height=input_height,
        first_conv=first_conv,
        maxpool1=maxpool1,
    )


def resnet50_encoder(first_conv: bool, *, maxpool1: bool, latent_dim: int) -> ResNetEncoder:
    return ResNetEncoder(
        EncoderBottleneck,
        layers=[3, 4, 6, 3],
        first_conv=first_conv,
        maxpool1=maxpool1,
        latent_dim=latent_dim,
    )


def resnet50_decoder(
    latent_dim: int, *, input_height: int, first_conv: bool, maxpool1: bool
) -> ResNetDecoder:
    return ResNetDecoder(
        DecoderBlock,
        layers=[3, 4, 6, 3],
        latent_dim=latent_dim,
        input_height=input_height,
        first_conv=first_conv,
        maxpool1=maxpool1,
    )


class ResNetVersion(Enum):
    RN18 = "18"
    RN50 = "50"


@dataclass
class ResNetAE(AeFactory):
    latent_dim: int = 128
    version: ResNetVersion = ResNetVersion.RN18
    first_conv: bool = False
    maxpool1: bool = False

    @implements(AeFactory)
    def __call__(
        self, input_shape: tuple[int, int, int]
    ) -> AeFactoryOut[ResNetEncoder, ResNetDecoder]:
        if self.version is ResNetVersion.RN18:
            enc_fn = resnet18_encoder
            dec_fn = resnet18_decoder
        else:
            enc_fn = resnet50_encoder
            dec_fn = resnet50_decoder
        encoder = enc_fn(
            self.first_conv,
            latent_dim=self.latent_dim,
            maxpool1=self.maxpool1,
        )
        decoder = dec_fn(
            self.latent_dim,
            input_height=input_shape[1],
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
        )
        return encoder, decoder, self.latent_dim
