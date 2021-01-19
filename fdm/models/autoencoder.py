from typing import Any, Dict, List, Optional, Tuple, Union, overload

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

from shared.configs import VaeStd
from shared.utils import sample_concrete, to_discrete

from .base import EncodingSize, ModelBase, Reconstructions, SplitEncoding

__all__ = ["AutoEncoder", "VAE"]


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: Optional[EncodingSize],
        use_amp: bool = True,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.encoding_size = encoding_size
        self.feature_group_slices = feature_group_slices
        self.use_amp = use_amp

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
        zs, zy = self.split_encoding(z)
        just_s = torch.cat([zs, torch.zeros_like(zy)], dim=1)
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

<<<<<<< HEAD
    def step(self, grad_scaler: Optional[GradScaler] = None):
        self.encoder.step(grad_scaler=grad_scaler)
        self.decoder.step(grad_scaler=grad_scaler)
=======
    def step(self):
        if self.use_amp:
            self.scaler.step(self.encoder.optimizer)
            self.scaler.step(self.decoder.optimizer)
            self.scaler.update()
        else:
            self.encoder.step()
            self.decoder.step()
>>>>>>> 80afebeacd2f04137b86883130e829ec0428bcaa

    def split_encoding(self, z: Tensor) -> SplitEncoding:
        assert self.encoding_size is not None
        zs, zy = z.split((self.encoding_size.zs, self.encoding_size.zy), dim=1)
        return SplitEncoding(zs=zs, zy=zy)

    def mask(self, z: Tensor, random: bool = False) -> Tuple[Tensor, Tensor]:
        """Split the encoding and mask out zs and zy. This is a cheap function."""
        zs, zy = self.split_encoding(z)
        if random:
            zs_m = torch.cat([torch.randn_like(zs), zy], dim=1)
            zy_m = torch.cat([zs, torch.randn_like(zy)], dim=1)
        else:
            zs_m = torch.cat([torch.zeros_like(zs), zy], dim=1)
            zy_m = torch.cat([zs, torch.zeros_like(zy)], dim=1)
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
        encoding = self.encode(x)

        recon_all = self.decode(encoding)
        recon_loss = recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        prior_loss = kl_weight * encoding.norm(dim=1).mean()
        loss = recon_loss + prior_loss
        return encoding, loss, {"Loss reconstruction": recon_loss.item(), "Prior Loss": prior_loss}


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: Optional[EncodingSize],
        vae_std_tform: VaeStd,
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
        self.vae_std_tform = vae_std_tform

    def compute_divergence(self, sample: Tensor, posterior: td.Distribution) -> Tensor:
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        return (log_q - log_p).sum()

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
            if self.vae_std_tform == VaeStd.softplus:
                scale = F.softplus(scale)
            else:
                scale = torch.exp(0.5 * scale).clamp(min=0.005, max=3.0)
            posterior = self.posterior_fn(loc, scale)
        sample = posterior.rsample() if stochastic else loc

        if return_posterior:
            return sample, posterior
        else:
            return sample

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
