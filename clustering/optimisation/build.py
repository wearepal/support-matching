from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from clustering.models import VAE, AutoEncoder
from shared.configs import RL, Config, Enc
from shared.models.configs import conv_autoencoder, fc_autoencoder

from .loss import MixedLoss, PixelCrossEntropy, VGGLoss

__all__ = ["build_ae"]


def build_ae(
    cfg: Config,
    input_shape: Tuple[int, ...],
    feature_group_slices: Optional[Dict[str, List[slice]]],
) -> Tuple[AutoEncoder, Tuple[int, ...]]:
    is_image_data = len(input_shape) > 2
    variational = cfg.clust.encoder == Enc.vae
    enc_shape: Tuple[int, ...]
    if is_image_data:
        decoding_dim = input_shape[0] * 256 if cfg.enc.recon_loss == RL.ce else input_shape[0]
        # if cfg.enc.recon_loss == "ce":
        decoder_out_act = None
        # else:
        #     decoder_out_act = nn.Sigmoid() if cfg.enc.dataset == "cmnist" else nn.Tanh()
        encoder, decoder, enc_shape = conv_autoencoder(
            input_shape,
            cfg.enc.init_chans,
            encoding_dim=cfg.enc.out_dim,
            decoding_dim=decoding_dim,
            levels=cfg.enc.levels,
            decoder_out_act=decoder_out_act,
            variational=variational,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            input_shape,
            cfg.enc.init_chans,
            encoding_dim=cfg.enc.out_dim,
            levels=cfg.enc.levels,
            variational=variational,
        )

    recon_loss_fn_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if cfg.enc.recon_loss == RL.l1:
        recon_loss_fn_ = nn.L1Loss(reduction="sum")
    elif cfg.enc.recon_loss == RL.l2:
        recon_loss_fn_ = nn.MSELoss(reduction="sum")
    elif cfg.enc.recon_loss == RL.bce:
        recon_loss_fn_ = nn.BCELoss(reduction="sum")
    elif cfg.enc.recon_loss == RL.huber:
        recon_loss_fn_ = lambda x, y: 0.1 * F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
    elif cfg.enc.recon_loss == RL.ce:
        recon_loss_fn_ = PixelCrossEntropy(reduction="sum")
    elif cfg.enc.recon_loss == RL.mixed:
        assert feature_group_slices is not None, "can only do multi gen_loss with feature groups"
        recon_loss_fn_ = MixedLoss(feature_group_slices, reduction="sum")
    else:
        raise ValueError(f"{cfg.enc.recon_loss} is an invalid reconstruction gen_loss")

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if cfg.clust.vgg_weight != 0:
        vgg_loss = VGGLoss()
        vgg_loss.to(cfg.misc._device)

        def recon_loss_fn(input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return recon_loss_fn_(input_, target) + cfg.clust.vgg_weight * vgg_loss(input_, target)

    else:
        recon_loss_fn = recon_loss_fn_
    optimizer_args = {"lr": cfg.clust.enc_lr, "weight_decay": cfg.clust.enc_wd}
    generator: AutoEncoder
    if variational:
        generator = VAE(
            encoder=encoder,
            decoder=decoder,
            recon_loss_fn=recon_loss_fn,
            kl_weight=cfg.clust.kl_weight,
            vae_std_tform=cfg.clust.vae_std_tform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        generator = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            recon_loss_fn=recon_loss_fn,
            kl_weight=cfg.clust.kl_weight,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    generator.to(cfg.misc._device)
    return generator, enc_shape
