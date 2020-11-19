"""Autoencoders"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.configs import VaeStd
from shared.utils import print_metrics, to_discrete, wandb_log

from .base import Encoder, ModelBase

__all__ = ["AutoEncoder", "VAE"]

log = logging.getLogger(__name__.split(".")[-1].upper())


class AutoEncoder(Encoder):
    """Classical AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor],
        kl_weight: float,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.recon_loss_fn = recon_loss_fn
        self.feature_group_slices = feature_group_slices
        self.prior_weight = kl_weight

    def encode(self, x: Tensor, stochastic: bool = False) -> Tensor:
        del stochastic
        return self.encoder(x)

    def decode(self, z: Tensor, discretize: bool = False) -> Tensor:
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
        logging_dict = {}
        # enc_sched = torch.optim.lr_scheduler.StepLR(self.encoder.optimizer, step_size=9, gamma=.3)
        # dec_sched = torch.optim.lr_scheduler.StepLR(self.decoder.optimizer, step_size=9, gamma=.3)
        with tqdm(total=epochs * len(train_data)) as pbar:
            for _ in range(epochs):

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    _, loss, logging_dict = self.routine(x)

                    loss.backward()
                    self.step()

                    enc_loss: float = loss.item()
                    pbar.update()
                    pbar.set_postfix(AE_loss=enc_loss)
                    if use_wandb:
                        step += 1
                        logging_dict.update({"Total Loss": enc_loss})
                        wandb_log(True, logging_dict, step)
                # enc_sched.step()
                # dec_sched.step()
        log.info("Final result from encoder training:")
        print_metrics({f"Enc {k}": v for k, v in logging_dict.items()})

    def routine(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        encoding = self.encode(x)
        recon_all = self.decode(encoding)
        recon_loss = self.recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        prior_loss = self.prior_weight * encoding.norm(dim=1).mean()
        loss = recon_loss + prior_loss
        logging_dict = {"Loss reconstruction": recon_loss.item(), "Prior Loss": prior_loss.item()}
        return encoding, loss, logging_dict

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
        vae_std_tform: VaeStd,
        feature_group_slices: Optional[Dict[str, List[slice]]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            recon_loss_fn=recon_loss_fn,
            kl_weight=kl_weight,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal
        self.vae_std_tform = vae_std_tform
        self.kl_weight = kl_weight

    def compute_divergence(self, sample: Tensor, posterior: td.Distribution) -> Tensor:
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        return (log_q - log_p).sum()

    def encode_with_posterior(self, x: Tensor) -> Tuple[Tensor, td.Distribution]:
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if self.vae_std_tform == VaeStd.softplus:
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
