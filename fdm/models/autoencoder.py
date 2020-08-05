from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from shared.utils import to_discrete, sample_concrete

from .base import ModelBase, EncodingSize, SplitEncoding, Reconstructions

__all__ = ["AutoEncoder", "VAE"]


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: Optional[EncodingSize],
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.encoding_size = encoding_size
        self.feature_group_slices = feature_group_slices

    def encode(self, inputs: Tensor, stochastic: bool = False) -> Tensor:
        del stochastic
        return self.encoder(inputs)

    def decode(self, z: Tensor, mode: Literal["soft", "hard", "relaxed"] = "soft") -> Tensor:
        decoding = self.decoder(z)
        if decoding.dim() == 4:
            # if decoding.size(1) <= 3:
            #     decoding = decoding.sigmoid()
            # else:
            if decoding.size(1) > 3:  # if we use CE losss, we have more than 3 channels
                # conversion for cross-entropy loss
                num_classes = 256
                decoding = decoding.view(decoding.size(0), num_classes, -1, *decoding.shape[-2:])
        else:
            if mode in ("hard", "relaxed") and self.feature_group_slices:
                discrete_outputs = []
                stop_index = 0
                #   Sample from discrete variables using the straight-through-estimator
                for group_slice in self.feature_group_slices["discrete"]:
                    if mode == "hard":
                        discrete_outputs.append(to_discrete(decoding[:, group_slice]).float())
                    else:
                        discrete_outputs.append(
                            sample_concrete(decoding[:, group_slice], temperature=1e-2)
                        )
                    stop_index = group_slice.stop
                discrete_outputs = torch.cat(discrete_outputs, axis=1)
                decoding = torch.cat([discrete_outputs, decoding[:, stop_index:]], axis=1)

        return decoding

    def all_recons(self, z: Tensor, mode: Literal["soft", "hard", "relaxed"]) -> Reconstructions:
        rand_s, rand_y = self.mask(z, random=True)
        zero_s, zero_y = self.mask(z)
        zs, zy, zn = self.split_encoding(z)
        just_s = torch.cat([zs, torch.zeros_like(zy), torch.zeros_like(zn)], dim=1)
        return Reconstructions(
            all=self.decode(z, mode=mode),
            rand_s=self.decode(rand_s, mode=mode),
            rand_y=self.decode(rand_y, mode=mode),
            zero_s=self.decode(zero_s, mode=mode),
            zero_y=self.decode(zero_y, mode=mode),
            just_s=self.decode(just_s, mode=mode),
        )

    def forward(self, inputs):
        return self.encode(inputs)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self):
        self.encoder.step()
        self.decoder.step()

    def split_encoding(self, z: Tensor) -> SplitEncoding:
        assert self.encoding_size is not None
        zs, zy, zn = z.split(
            (self.encoding_size.zs, self.encoding_size.zy, self.encoding_size.zn), dim=1
        )
        return SplitEncoding(zs=zs, zy=zy, zn=zn)

    def mask(self, z: Tensor, random: bool = False) -> Tuple[Tensor, Tensor]:
        """Split the encoding and mask out zs and zy. This is a cheap function."""
        zs, zy, zn = self.split_encoding(z)
        if random:
            # the question here is whether to have one random number per sample
            # or whether to also have distinct random numbers for all the dimensions of zs.
            # if we don't expect s to be complicated, then the former should suffice
            rand_zs = torch.randn((zs.size(0),) + (zs.dim() - 1) * (1,), device=zs.device)
            zs_m = torch.cat([rand_zs + torch.zeros_like(zs), zy, zn], dim=1)
            rand_zy = torch.randn((zy.size(0),) + (zy.dim() - 1) * (1,), device=zy.device)
            zy_m = torch.cat([zs, rand_zy + torch.zeros_like(zy), zn], dim=1)
        else:
            zs_m = torch.cat([torch.zeros_like(zs), zy, zn], dim=1)
            zy_m = torch.cat([zs, torch.zeros_like(zy), zn], dim=1)
        return zs_m, zy_m

    def fit(self, train_data: DataLoader, epochs: int, device, loss_fn, kl_weight: float):
        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for _ in range(epochs):

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    _, loss, _ = self.routine(x, recon_loss_fn=loss_fn, kl_weight=kl_weight)
                    # loss /= x[0].nelement()

                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())

    def routine(
        self, x: Tensor, recon_loss_fn, kl_weight: float
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        del kl_weight
        encoding = self.encode(x)

        recon_all = self.decode(encoding)
        recon_loss = recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        return encoding, recon_loss, {"Loss reconstruction": recon_loss.item()}


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: Optional[EncodingSize],
        std_transform: Literal["softplus", "exp"],
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
        self.std_transform = std_transform

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
        self, x: Tensor, return_posterior: bool = False, stochastic: bool = False
    ) -> Union[Tuple[Tensor, td.Distribution], Tensor]:
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if stochastic or return_posterior:
            if self.std_transform == "softplus":
                scale = F.softplus(scale)
            else:
                scale = torch.exp(0.5 * scale).clamp(min=0.005, max=3.0)
            posterior = self.posterior_fn(loc, scale)
        sample = posterior.rsample() if stochastic else loc

        if return_posterior:
            return sample, posterior
        else:
            return sample

    # def mask(self, z: Tensor, random: bool = False) -> Tuple[Tensor, Tensor]:
    #     return super().mask(z, random=False)

    def routine(
        self, x: Tensor, recon_loss_fn, kl_weight: float
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        encoding, posterior = self.encode(x, return_posterior=True, stochastic=True)
        kl_div = self.compute_divergence(encoding, posterior)
        kl_div /= x.nelement()
        kl_div *= kl_weight

        recon_all = self.decode(encoding)
        recon_loss = recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        elbo = recon_loss + kl_div
        logging_dict = {"Loss Reconstruction": recon_loss.item(), "KL divergence": kl_div}
        return encoding, elbo, logging_dict
