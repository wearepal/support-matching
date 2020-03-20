from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union, overload

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fdm.utils import to_discrete

from .base import ModelBase

__all__ = ["AutoEncoder", "VAE", "EncodingSize", "SplitEncoding", "Reconstructions"]


class EncodingSize(NamedTuple):
    zs: int
    zy: int
    zn: int


class SplitEncoding(NamedTuple):
    zs: Tensor
    zy: Tensor
    zn: Tensor


class Reconstructions(NamedTuple):
    all: Tensor
    rand_s: Tensor  # reconstruction with random s
    rand_y: Tensor  # reconstruction with random y


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: EncodingSize,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.encoding_size = encoding_size
        self.feature_group_slices = feature_group_slices

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def encode_and_split(self, inputs: Tensor) -> SplitEncoding:
        return self.split_encoding(self.encode(inputs))

    def reconstruct(self, encoding, s=None):
        decoding = self.decode(encoding, s)

        if decoding.dim() == 4:
            if decoding.size(1) > 3:
                # conversion for cross-entropy loss
                num_classes = 256
                decoding = decoding[:64].view(decoding.size(0), num_classes, -1, *decoding.shape[-2:])
                fac = num_classes - 1
                # `.max` also returns the index (i.e. argmax) which is what we want here
                decoding = decoding.max(dim=1)[1].float() / fac

        return decoding

    def decode(self, z, discretize: bool = False):
        decoding = self.decoder(z)
        
        if decoding.dim() == 4 and decoding.size(1) <= 3:
            decoding = decoding.sigmoid()

        if discretize and self.feature_group_slices:
            for group_slice in self.feature_group_slices["discrete"]:
                one_hot = to_discrete(decoding[:, group_slice])
                decoding[:, group_slice] = one_hot

        return decoding

    def generate_recon_rand_s(self, inputs: Tensor) -> Tensor:
        zs, zy, zn = self.encode_and_split(inputs)
        zs_m = torch.cat([torch.randn_like(zs), zy, zn], dim=1)
        return self.decode(zs_m)

    def decode_and_mask(self, z: Tensor, discretize: bool = False) -> Reconstructions:
        zs_m, zy_m = self.random_mask(z)
        recon_all = self.decode(z, discretize=discretize)
        recon_rand_s = self.decode(zs_m, discretize=discretize)
        recon_rand_y = self.decode(zy_m, discretize=discretize)
        return Reconstructions(all=recon_all, rand_s=recon_rand_s, rand_y=recon_rand_y)

    def forward(self, inputs, reverse: bool = True):
        if reverse:
            return self.decode(inputs)
        else:
            return self.encode(inputs)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self):
        self.encoder.step()
        self.decoder.step()

    def split_encoding(self, z: Tensor) -> SplitEncoding:
        zs, zy, zn = z.split(
            (self.encoding_size.zs, self.encoding_size.zy, self.encoding_size.zn), dim=1
        )
        return SplitEncoding(zs=zs, zy=zy, zn=zn)

    def random_mask(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        zs, zy, zn = self.split_encoding(z)
        zs_m = torch.cat([torch.randn_like(zs), zy, zn], dim=1)
        zy_m = torch.cat([zs, torch.randn_like(zy), zn], dim=1)
        return zs_m, zy_m


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: EncodingSize,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal

    def compute_divergence(self, sample: Tensor, posterior: td.Distribution) -> Tensor:
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        kl_div = (log_q - log_p).sum()

        return kl_div

    @overload
    def encode(
        self, x: Tensor, return_posterior: Literal[True], stochastic: bool = ...
    ) -> Tuple[Tensor, td.Distribution]:
        ...

    @overload
    def encode(
        self, x: Tensor, return_posterior: Literal[False] = ..., stochastic: bool = ...
    ) -> Tensor:
        ...

    def encode(
        self, x: Tensor, return_posterior: bool = False, stochastic: bool = True
    ) -> Union[Tuple[Tensor, td.Distribution], Tensor]:
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if stochastic or return_posterior:
            scale = F.softplus(scale)
            posterior = self.posterior_fn(loc, scale)
        sample = posterior.rsample() if stochastic else loc

        if return_posterior:
            return sample, posterior
        else:
            return sample
