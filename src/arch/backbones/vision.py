from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import override

import timm
import torch.nn as nn
import torchvision.models as tvm

from src.arch.common import Activation, BiaslessLayerNorm

from .base import BackboneFactory, BackboneFactoryOut

if TYPE_CHECKING:
    import timm.models as tm

__all__ = ["Beit", "ConvNeXt", "DenseNet", "NfNet", "ResNet", "SimpleCNN", "Swin", "SwinV2", "ViT"]


class ResNetVersion(Enum):
    RN18 = "18"
    RN34 = "34"
    RN50 = "50"
    RN101 = "101"


@dataclass
class ResNet(BackboneFactory):
    pretrained: bool = False
    version: ResNetVersion = ResNetVersion.RN18

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut[tvm.ResNet]:
        fn_name = f"resnet{self.version.value}"
        if self.pretrained:
            weights_enum_name = f"ResNet{self.version.value}_Weights"
            weights = getattr(tvm, weights_enum_name).DEFAULT
        else:
            weights = None
        model: tvm.ResNet = getattr(tvm, fn_name)(weights=weights)
        out_dim = model.fc.in_features
        model.fc = cast(nn.Linear, nn.Identity())
        return model, out_dim


class DenseNetVersion(Enum):
    DN121 = "121"
    DN161 = "161"
    DN169 = "169"
    DN201 = "201"


@dataclass
class DenseNet(BackboneFactory):
    pretrained: bool = False
    version: DenseNetVersion = DenseNetVersion.DN121

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut[tvm.DenseNet]:
        fn_name = f"densenet{self.version.value}"
        if self.pretrained:
            weights_enum_name = f"DenseNet{self.version.value}_Weights"
            weights = getattr(tvm, weights_enum_name).DEFAULT
        else:
            weights = None
        model: tvm.DenseNet = getattr(tvm, fn_name)(weights=weights)
        out_dim = model.classifier.in_features
        model.classifier = cast(nn.Linear, nn.Identity())
        return model, out_dim


class ConvNeXtVersion(Enum):
    TINY = "convnext_tiny"
    SMALL = "convnext_small"
    BASE = "convnext_base"
    BASE_21K = "convnext_base_in22k"
    LARGE = "convnext_large"
    LARGE_21K = "convnext_large_in22k"
    XLARGE_21K = "convnext_xlarge_in22k"


@dataclass
class ConvNeXt(BackboneFactory):
    pretrained: bool = False
    version: ConvNeXtVersion = ConvNeXtVersion.BASE
    checkpoint_path: str = ""
    p: float = 3

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut[nn.Sequential]:
        classifier: "tm.ConvNeXt" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        out_dim = classifier.num_features
        pooling_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        backbone = nn.Sequential(
            classifier.stem,
            classifier.stages,
            classifier.norm_pre,
            pooling_module,
        )
        return backbone, out_dim


class ViTVersion(Enum):
    TINY_P16_224 = "vit_tiny_patch16_224"
    TINY_P16_224_21K = "vit_tiny_patch16_224_in21k"
    TINY_P16_384 = "vit_tiny_patch16_384"

    SMALL_P16_224_21K = "vit_small_patch16_224_in21k"
    SMALL_P32_224_21K = "vit_small_patch32_224_in21k"
    SMALL_P16_384 = "vit_small_patch16_384"
    SMALL_P32_384 = "vit_small_patch32_384"

    BASE_P8_224_21K = "vit_base_patch8_224_in21k"
    BASE_P16_224_21K = "vit_base_patch16_224_in21k"
    BASE_P16_384 = "vit_base_patch16_384"
    BASE_P32_384 = "vit_base_patch32_384"

    LARGE_P32_224_21K = "vit_large_patch32_224_in21k"
    LARGE_P16_224_21K = "vit_large_patch16_224_in21k"
    LARGE_P16_384 = "vit_large_patch16_384"
    LARGE_P32_384 = "vit_large_patch32_384"

    HUGE_P14_224_21K = "vit_huge_patch14_224_in21k"


@dataclass
class ViT(BackboneFactory):
    pretrained: bool = True
    version: ViTVersion = ViTVersion.BASE_P16_224_21K
    checkpoint_path: str = ""

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut["tm.VisionTransformer"]:
        model: "tm.VisionTransformer" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Identity()
        return model, model.num_features


class SwinVersion(Enum):
    BASE_P4_W7_224_21K = "swin_base_patch4_window7_224_in22k"
    BASE_P4_W12_384_21K = "swin_base_patch4_window12_384_in22k"
    LARGE_P4_W12_224_21K = "swin_large_patch4_window12_224_in22k"
    LARGE_P4_W12_384_21K = "swin_large_patch4_window12_384_in22k"


@dataclass
class Swin(BackboneFactory):
    pretrained: bool = True
    version: SwinVersion = SwinVersion.BASE_P4_W7_224_21K
    checkpoint_path: str = ""

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut["tm.SwinTransformer"]:
        model: "tm.SwinTransformer" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Identity()  # type: ignore
        return model, model.num_features


class SwinV2Version(Enum):
    BASE_W8_256 = "swinv2_base_window8_256"
    BASE_W12_196 = "swinv2_base_window12_192_22k"
    BASE_W16_256 = "swinv2_base_window16_256"
    BASE_W12TO16_192TO256 = "swinv2_base_window16to_192to256_22kft1k"
    BASE_W12TO24_192TO384 = "swinv2_base_window12to24_192to384_22kft1k"
    LARGE_W12_192 = "swinv2_large_window12_192_22k"
    LARGE_W12TO16_192TO256 = "swinv2_large_window12to16_192to256_22kft1k"
    LARGE_W12TO24_192TO384 = "swinv2_large_window12to24_192to384_22kft1k"

    CR_BASE_224 = "swinv2_cr_base_224"
    CR_BASE_384 = "swinv2_cr_base_384"
    CR_LARGE_224 = "swinv2_cr_large_224"
    CR_LARGE_384 = "swinv2_cr_large_384"
    CR_HUGE_224 = "swinv2_cr_huge_224"
    CR_HUGE_384 = "swinv2_cr_huge_384"
    CR_GIANT_224 = "swinv2_cr_giant_224"
    CR_GIANT_384 = "swinv2_cr_giant_384"


@dataclass
class SwinV2(BackboneFactory):
    pretrained: bool = True
    version: SwinV2Version = SwinV2Version.BASE_W8_256
    checkpoint_path: str = ""
    freeze_patch_embedder: bool = True
    out_dim: int = 0

    @override
    def __call__(
        self, input_dim: int
    ) -> BackboneFactoryOut[Union["tm.SwinTransformerV2", "tm.SwinTransformerV2Cr"]]:
        model: "tm.SwinTransformerV2 | tm.SwinTransformerV2Cr" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        if self.freeze_patch_embedder:
            for param in model.patch_embed.parameters():
                param.requires_grad_(False)
        model.reset_classifier(num_classes=self.out_dim)
        out_dim = self.out_dim if self.out_dim > 0 else model.num_features
        return model, out_dim


class BeitVersion(Enum):
    BASE_P16_224 = "beit_base_patch16_224"
    BASE_P16_224_21K = "beit_base_patch16_224_in22k"
    BASE_P16_384 = "beit_base_patch16_384"
    LARGE_P16_224_21K = "beit_large_patch16_224_in22k"
    LARGE_P16_384 = "beit_large_patch16_384"
    LARGE_P16_512 = "beit_large_patch16_512"


@dataclass
class Beit(BackboneFactory):
    pretrained: bool = True
    version: BeitVersion = BeitVersion.BASE_P16_224_21K
    checkpoint_path: str = ""
    out_dim: int = 0

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut["tm.Beit"]:
        model: "tm.Beit" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.reset_classifier(num_classes=self.out_dim)
        out_dim = self.out_dim if self.out_dim > 0 else model.num_features
        return model, out_dim


class NfNetVersion(Enum):
    SE_RN26 = "nf_seresnet26"
    SE_RN50 = "nf_seresnet50"
    SE_RN101 = "nf_seresnet101"
    EC_RN26 = "nf_ecaresnet26"
    EC_RN50 = "nf_ecaresnet50"
    EC_RN101 = "nf_ecaresnet101"
    RN50 = "nf_resnet50"
    RN26 = "nf_resnet26"
    RN101 = "nf_resnet101"
    F4 = "nfnet_f4"
    F5 = "nfnet_f5"
    F6 = "nfnet_f6"
    DM_F4 = "dm_nfnet_f4"
    DM_F6 = "dm_nfnet_f6"
    DM_F5 = "dm_nfnet_f5"


@dataclass
class NfNet(BackboneFactory):
    pretrained: bool = True
    version: NfNetVersion = NfNetVersion.F6
    checkpoint_path: str = ""
    out_dim: int = 0

    @override
    def __call__(self, input_dim: int) -> BackboneFactoryOut["tm.NormFreeNet"]:
        model: "tm.NormFreeNet" = timm.create_model(
            self.version.value, pretrained=self.pretrained, checkpoint_path=self.checkpoint_path
        )
        model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())  # type: ignore
        model.reset_classifier(num_classes=self.out_dim)
        out_dim = self.out_dim if self.out_dim > 0 else model.num_features
        return model, out_dim


class DownsamplingOp(Enum):
    MP = "MaxPool"
    STRIDE = "Stride"


class NormType(Enum):
    BN = (nn.BatchNorm1d,)
    LN = (BiaslessLayerNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init


@dataclass
class SimpleCNN(BackboneFactory):
    norm: NormType | None = NormType.BN
    activation: Activation = Activation.GELU
    levels: int = 4
    blocks_per_level: int = 1
    c0: int = 64
    ds_op: DownsamplingOp = DownsamplingOp.MP

    def _conv_block(
        self,
        in_dim: int,
        *,
        out_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = "same",
    ) -> nn.Sequential:
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.norm is not None:
            _block.append(self.norm.init(out_dim))
        _block.append(self.activation.init())
        return nn.Sequential(*_block)

    def __call__(self, input_dim: int) -> BackboneFactoryOut[nn.Sequential]:
        layers = []
        for i in range(self.levels):
            out_dim = self.c0**i
            for j in range(self.blocks_per_level - 1):
                if (self.ds_op is DownsamplingOp.STRIDE) and (j == (self.blocks_per_level - 1)):
                    stride = 2
                    padding = 0
                else:
                    stride = 1
                    padding = "same"
                layers.append(
                    self._conv_block(
                        input_dim, out_dim=out_dim, kernel_size=3, stride=stride, padding=padding
                    )
                )
                if j == 0:
                    input_dim = out_dim
            if self.ds_op is DownsamplingOp.MP:
                layers.append(nn.MaxPool2d(2, 2))

        layers.append(nn.AdaptiveAvgPool2d(1))

        return nn.Sequential(*layers), input_dim
