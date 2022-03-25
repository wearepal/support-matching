from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Sequence, Tuple, cast
from typing_extensions import Literal, Self

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from advrep.models.base import Model
from advrep.optimisation.loss import MixedLoss
from shared.utils import RoundSTE, sample_concrete, to_discrete

__all__ = [
    "AutoEncoder",
    "EncodingSize",
    "Model",
    "Reconstructions",
    "SplitEncoding",
]


@dataclass
class EncodingSize:
    zs: int
    zy: int


@dataclass
class SplitEncoding:
    zs: Tensor
    zy: Tensor

    def join(self) -> Tensor:
        return torch.cat([self.zs, self.zy], dim=1)

    @property
    def card_y(self) -> int:
        return self.zy.size(1)

    @property
    def card_s(self) -> int:
        return self.zs.size(1)

    def mask(self, random: bool = False, *, detach: bool = False) -> Tuple[Self, Self]:
        """Mask out zs and zy. This is a cheap function.

        :param enc: encoding to mask
        :param random: whether to replace the masked out part with random noise
        :param detach: whether to detach from the computational graph before masking
        """
        zs = self.zs
        zy = self.zy
        if detach:
            zs = zs.detach()
            zy = zy.detach()
        if random:
            if zs.size(1) > 1:
                random_s = torch.randint(
                    low=0, high=self.card_s, size=(zs.size(0),), device=zs.device
                )
                random_zs = F.one_hot(random_s, num_classes=self.card_s)
            else:
                random_zs = torch.randint_like(zs, low=0, high=2)
            zs_m = SplitEncoding(zs=random_zs.float(), zy=zy)
            zy_m = SplitEncoding(zs=zs, zy=torch.randn_like(zy))
        else:
            zs_m = SplitEncoding(zs=torch.zeros_like(zs), zy=zy)
            zy_m = SplitEncoding(zs=zs, zy=torch.zeros_like(zy))
        return zs_m, zy_m


@dataclass
class Reconstructions:
    all: Tensor
    rand_s: Tensor  # reconstruction with random s
    rand_y: Tensor  # reconstruction with random y
    zero_s: Tensor
    zero_y: Tensor
    just_s: Tensor


class ReconstructionLoss(Enum):
    """Reconstruction loss."""

    l1 = nn.L1Loss
    l2 = nn.MSELoss
    bce = nn.BCELoss
    huber = nn.SmoothL1Loss
    mixed = MixedLoss


class ZsTransform(Enum):
    """How to transform the z_s partition."""

    none = auto()
    round_ste = auto()


class AutoEncoder(Model):
    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        latent_dim: int,
        zs_dim: int,
        zs_transform: ZsTransform = ZsTransform.none,
        feature_group_slices: dict[str, list[slice]] | None = None,
        lr: float = 5.0e-4,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: dict[str, Any] | None = None,
        recon_loss_fn: ReconstructionLoss = ReconstructionLoss.l2,
    ) -> None:

        self.latent_dim = latent_dim
        self.encoding_size = EncodingSize(zs=zs_dim, zy=latent_dim - zs_dim)
        self.feature_group_slices = feature_group_slices
        self.zs_transform = zs_transform

        encoder, decoder = self._build(input_shape)
        model = nn.Sequential(encoder, decoder)
        super().__init__(
            model=model, lr=lr, optimizer_cls=optimizer_cls, optimizer_kwargs=optimizer_kwargs
        )
        self.encoder = encoder
        self.decoder = decoder

        if recon_loss_fn is ReconstructionLoss.mixed:
            if feature_group_slices is None:
                raise ValueError("'MixedLoss' requires 'feature_group_slices' to be specified.")
            self.recon_loss_fn = recon_loss_fn.value(
                reduction="sum", feature_group_slices=feature_group_slices
            )
        else:
            self.recon_loss_fn = recon_loss_fn.value(reduction="sum")

    @abstractmethod
    def _build(self, input_shape: Sequence[int]) -> Tuple[nn.Module, nn.Module]:
        ...

    def encode(self, inputs: Tensor, *, transform_zs: bool = True) -> SplitEncoding:
        enc = self._split_encoding(self.encoder(inputs))
        if transform_zs and self.zs_transform is ZsTransform.round_ste:
            rounded_zs = RoundSTE.apply(torch.sigmoid(enc.zs))
        else:
            rounded_zs = enc.zs
        return SplitEncoding(zs=rounded_zs, zy=enc.zy)

    def decode(
        self,
        split_encoding: SplitEncoding,
        *,
        s: Tensor | None = None,
        mode: Literal["soft", "hard", "relaxed"] = "soft",
    ) -> Tensor:
        if s is not None:  # we've been given the ground-truth labels for reconstruction
            card_s = split_encoding.zy.size(1)
            if card_s > 1:
                s = cast(Tensor, F.one_hot(s.long(), num_classes=card_s))
            else:
                s = s.view(-1, 1)
            split_encoding = replace(split_encoding, zs=s.float())

        decoding = self.decoder(split_encoding.join())
        if mode in ("hard", "relaxed") and self.feature_group_slices:
            discrete_outputs_ls: list[Tensor] = []
            stop_index = 0
            #   Sample from discrete variables using the straight-through-estimator
            for group_slice in self.feature_group_slices["discrete"]:
                if mode == "hard":
                    discrete_outputs_ls.append(to_discrete(decoding[:, group_slice]).float())
                else:
                    discrete_outputs_ls.append(
                        sample_concrete(decoding[:, group_slice], temperature=1e-2)
                    )
                stop_index = group_slice.stop
            discrete_outputs = torch.cat(discrete_outputs_ls, dim=1)
            decoding = torch.cat([discrete_outputs, decoding[:, stop_index:]], dim=1)

        return decoding

    def all_recons(
        self, enc: SplitEncoding, mode: Literal["soft", "hard", "relaxed"]
    ) -> Reconstructions:
        rand_s, rand_y = enc.mask(random=True)
        zero_s, zero_y = enc.mask()
        just_s = SplitEncoding(zs=enc.zs, zy=torch.zeros_like(enc.zy))
        return Reconstructions(
            all=self.decode(enc, mode=mode),
            rand_s=self.decode(rand_s, mode=mode),
            rand_y=self.decode(rand_y, mode=mode),
            zero_s=self.decode(zero_s, mode=mode),
            zero_y=self.decode(zero_y, mode=mode),
            just_s=self.decode(just_s, mode=mode),
        )

    def _split_encoding(self, z: Tensor) -> SplitEncoding:
        assert self.encoding_size is not None
        zs, zy = z.split((self.encoding_size.zs, self.encoding_size.zy), dim=1)
        return SplitEncoding(zs=zs, zy=zy)

    def training_step(
        self,
        x: Tensor,
        *,
        prior_loss_w: float,
        s: Tensor | None = None,
    ) -> tuple[SplitEncoding, Tensor, dict[str, float]]:
        # it only makes sense to transform zs if we're actually going to use it
        encoding = self.encode(x, transform_zs=s is None)
        recon_all = self.decode(encoding, s=s)

        recon_loss = self.recon_loss_fn(recon_all, x)
        recon_loss /= x.numel()
        prior_loss = prior_loss_w * encoding.zy.norm(dim=1).mean()
        loss = recon_loss + prior_loss

        logging_dict = {
            "Loss Reconstruction": recon_loss.detach().cpu().item(),
            "Prior Loss": prior_loss.detach().cpu().item(),
        }
        return encoding, loss, logging_dict

    def forward(self, inputs: Tensor) -> SplitEncoding:
        return self.encode(inputs)
