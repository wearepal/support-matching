"""Autoencoders"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Callable

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from shared.utils import to_discrete, wandb_log

from .base import ModelBase, Encoder

__all__ = ["AutoEncoder", "VAE"]


class AutoEncoder(Encoder):
    """Classical AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor],
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.recon_loss_fn = recon_loss_fn
        self.feature_group_slices = feature_group_slices

    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        del stochastic
        return self.encoder(x)

    def decode(self, z: Tensor, discretize: bool = False) -> Tensor:
       import pdb; pdb.set_trace()
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
            if discretize and self.feature_group_slices:
                for group_slice in self.feature_group_slices["discrete"]:
                    one_hot = to_discrete(decoding[:, group_slice])
                    decoding[:, group_slice] = one_hot

        return decoding

    def forward(self, inputs: Tensor, reverse: bool = False) -> Tensor:
        if reverse:
            return self.decode(inputs)
        else:
            return self.encode(inputs)

    def zero_grad(self) -> None:
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.encoder.step(grads)
        self.decoder.step(grads)

    def fit(
        self, train_data: DataLoader, epochs: int, device: torch.device, use_wandb: bool
    ) -> None:
        self.train()

        step = 0
        use_wandb_ = use_wandb

        class _Namespace:
            use_wandb: bool = use_wandb_

        args = _Namespace()
        with tqdm(total=epochs * len(train_data)) as pbar:
            for _ in range(epochs):

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    _, loss, _ = self.routine(x)
                    # loss /= x[0].nelement()

                    loss.backward()
                    self.step()

                    enc_loss = loss.detach().cpu().numpy()
                    pbar.update()
                    pbar.set_postfix(AE_loss=enc_loss)
                    step += 1
                    wandb_log(args, {"enc_loss": enc_loss}, step)

    def routine(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        encoding = self.encode(x)

        recon_all = self.decode(encoding)
        recon_loss = self.recon_loss_fn(recon_all, x)
        recon_loss /= x.size(0)
        return encoding, recon_loss, {"Loss reconstruction": recon_loss.item()}

    def freeze_initial_layers(self, num_layers: int, optimizer_kwargs: Dict[str, Any]) -> None:
        self.encoder.freeze_initial_layers(num_layers=num_layers, optimizer_kwargs=optimizer_kwargs)


class VAE(AutoEncoder):
    """Variational AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor],
        kl_weight: float,
        std_transform: Literal["softplus", "exp"],
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            recon_loss_fn=recon_loss_fn,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal
        self.std_transform = std_transform
        self.kl_weight = kl_weight

    def compute_divergence(self, sample: Tensor, posterior: td.Distribution) -> Tensor:
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        kl_div = (log_q - log_p).sum()

        return kl_div

    def encode_with_posterior(self, x: Tensor) -> Tuple[Tensor, td.Distribution]:
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if self.std_transform == "softplus":
            scale = F.softplus(scale)
        else:
            scale = torch.exp(0.5 * scale).clamp(min=0.005, max=3.0)
        posterior = self.posterior_fn(loc, scale)
        sample = posterior.rsample()

        return sample, posterior

    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        if stochastic:
            sample, _ = self.encode_with_posterior(x)
        else:
            loc, _ = self.encoder(x).chunk(2, dim=1)
            sample = loc
        return sample

    def routine(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        encoding, posterior = self.encode_with_posterior(x)
        kl_div = self.compute_divergence(encoding, posterior)
        kl_div /= x.size(0)
        kl_div *= self.kl_weight

        recon_all = self.decode(encoding)
        recon_loss = self.recon_loss_fn(recon_all, x)
        recon_loss /= x.size(0)
        elbo = recon_loss + kl_div
        logging_dict = {"Loss Reconstruction": recon_loss.item(), "KL divergence": kl_div}
        return encoding, elbo, logging_dict
