from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from fdm.models import (
    VAE,
    AutoEncoder,
    EncodingSize,
    PartitionedAeInn,
    build_conv_inn,
    build_fc_inn,
)
from shared.configs import Config

__all__ = ["build_inn", "build_ae"]


def build_inn(
    cfg: Config,
    autoencoder: AutoEncoder,
    ae_loss_fn,
    is_image_data: bool,
    save_dir: Path,
    ae_enc_shape: Tuple[int, ...],
    context_loader: DataLoader,
) -> PartitionedAeInn:
    inn_fn = build_conv_inn if is_image_data else build_fc_inn
    inn = PartitionedAeInn(
        args=cfg.fdm,
        optimizer_args={"lr": cfg.fdm.inn_lr, "weight_decay": cfg.fdm.weight_decay},
        input_shape=ae_enc_shape,
        autoencoder=autoencoder,
        model=inn_fn(cfg.fdm, ae_enc_shape),
    )
    inn.to(cfg.misc._device)

    if cfg.fdm.path_to_ae:
        save_dict = torch.load(cfg.fdm.path_to_ae, map_location=lambda storage, loc: storage)
        autoencoder.load_state_dict(save_dict["model"])
        if "args" in save_dict:
            args_ae = save_dict["args"]
            assert cfg.enc.init_chans == args_ae["init_channels"]
            assert cfg.enc.levels == args_ae["levels"]
    else:
        inn.fit_ae(
            context_loader,
            epochs=cfg.fdm.ae_epochs,
            device=cfg.misc._device,
            loss_fn=ae_loss_fn,
            kl_weight=cfg.fdm.kl_weight,
        )
        # the args names follow the convention of the standalone VAE commandline args
        args_ae = {"init_channels": cfg.enc.init_chans, "levels": cfg.enc.levels}
        torch.save({"model": autoencoder.state_dict(), "args": args_ae}, save_dir / "autoencoder")
    return inn


def build_ae(
    cfg: Config,
    encoder,
    decoder,
    encoding_size: Optional[EncodingSize],
    feature_group_slices: Optional[Dict[str, List[slice]]],
) -> AutoEncoder:
    optimizer_args = {"lr": cfg.fdm.lr, "weight_decay": cfg.fdm.weight_decay}
    generator: AutoEncoder
    if cfg.fdm.vae:
        generator = VAE(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            vae_std_tform=cfg.fdm.vae_std_tform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        generator = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    generator.to(cfg.misc._device)
    return generator
