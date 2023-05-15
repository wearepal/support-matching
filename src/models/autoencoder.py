from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from typing_extensions import Literal, Self, override

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.arch.autoencoder import AePair
from src.discrete import discretize, round_ste, sample_concrete
from src.loss import MixedLoss
from src.models.base import Model, ModelCfg
from src.utils import to_item

__all__ = [
    "SplitLatentAe",
    "SplitLatentAeCfg",
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


@dataclass
class SplitLatentAeCfg(ModelCfg):
    """These are the parameters to `SplitLatentAe` which are configurable by hydra."""

    zs_dim: Union[int, float] = 1
    zs_transform: ZsTransform = ZsTransform.none
    recon_loss: ReconstructionLoss = ReconstructionLoss.l2


@dataclass(repr=False, eq=False)
class SplitLatentAe(Model):
    model: AePair  # overriding the definition in `Model`
    cfg: SplitLatentAeCfg  # overriding the definition in `Model`
    feature_group_slices: Optional[Dict[str, List[slice]]] = None
    recon_loss_fn: Callable[[Tensor, Tensor], Tensor] = field(init=False)
    zs_dim: int = field(init=False)

    def __post_init__(self) -> None:
        zs_dim_t = self.cfg.zs_dim
        self.latent_dim: int = self.model.latent_dim
        self.zs_dim = round(zs_dim_t * self.latent_dim) if isinstance(zs_dim_t, float) else zs_dim_t
        self.encoding_size = EncodingSize(zs=self.zs_dim, zy=self.latent_dim - self.zs_dim)

        if self.cfg.recon_loss is ReconstructionLoss.mixed:
            if self.feature_group_slices is None:
                raise ValueError("'MixedLoss' requires 'feature_group_slices' to be specified.")
            self.recon_loss_fn = self.cfg.recon_loss.value(
                reduction="sum", feature_group_slices=self.feature_group_slices
            )
        else:
            self.recon_loss_fn = self.cfg.recon_loss.value(reduction="sum")
        super().__post_init__()

    def encode(self, inputs: Tensor, *, transform_zs: bool = True) -> SplitEncoding:
        enc = self._split_encoding(self.model.encoder(inputs))
        if transform_zs and self.cfg.zs_transform is ZsTransform.round_ste:
            rounded_zs = round_ste(torch.sigmoid(enc.zs))
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
            card_s = split_encoding.zs.size(1)
            if card_s > 1:
                s_ = cast(Tensor, F.one_hot(s.long(), num_classes=card_s))
            else:
                s_ = s.view(-1, 1)
            split_encoding = replace(split_encoding, zs=s_.float())

        decoding = self.model.decoder(split_encoding.join())
        if mode in ("hard", "relaxed") and self.feature_group_slices:
            discrete_outputs_ls: list[Tensor] = []
            stop_index = 0
            #   Sample from discrete variables using the straight-through-estimator
            for group_slice in self.feature_group_slices["discrete"]:
                if mode == "hard":
                    discrete_outputs_ls.append(discretize(decoding[:, group_slice]).float())
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
        s: Tensor | None = None,
        prior_loss_w: Optional[float] = None,
    ) -> tuple[SplitEncoding, Tensor, dict[str, float]]:
        # it only makes sense to transform zs if we're actually going to use it
        encoding = self.encode(x, transform_zs=s is None)
        recon_all = self.decode(encoding, s=s)

        loss = self.recon_loss_fn(recon_all, x)
        loss /= x.numel()
        logging_dict = {"loss/reconstruction": to_item(loss)}
        if (prior_loss_w is not None) and (prior_loss_w > 0.0):
            prior_loss = prior_loss_w * encoding.zy.norm(dim=1).mean()
            logging_dict["loss/prior"] = to_item(prior_loss)
            loss += prior_loss
        return encoding, loss, logging_dict

    @override
    def forward(self, inputs: Tensor) -> SplitEncoding:
        return self.encode(inputs)
