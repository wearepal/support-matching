from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from fdm.configs import VaeArgs
from fdm.models import (
    VAE,
    AutoEncoder,
    EncodingSize,
    PartitionedAeInn,
    build_conv_inn,
    build_fc_inn,
)

__all__ = ["build_inn", "build_ae"]


def build_inn(
    args: VaeArgs,
    autoencoder: AutoEncoder,
    ae_loss_fn,
    is_image_data: bool,
    save_dir: Path,
    ae_enc_shape: Tuple[int, ...],
    context_loader: DataLoader,
) -> PartitionedAeInn:
    inn_fn = build_conv_inn if is_image_data else build_fc_inn
    inn = PartitionedAeInn(
        args=args,
        optimizer_args={"lr": args.inn_lr, "weight_decay": args.weight_decay},
        input_shape=ae_enc_shape,
        autoencoder=autoencoder,
        model=inn_fn(args, ae_enc_shape),
    )
    inn.to(args._device)

    if args.path_to_ae:
        save_dict = torch.load(args.path_to_ae, map_location=lambda storage, loc: storage)
        autoencoder.load_state_dict(save_dict["model"])
        if "args" in save_dict:
            args_ae = save_dict["args"]
            assert args.init_channels == args_ae["init_channels"]
            assert args.enc_levels == args_ae["levels"]
    else:
        inn.fit_ae(
            context_loader,
            epochs=args.ae_epochs,
            device=args._device,
            loss_fn=ae_loss_fn,
            kl_weight=args.kl_weight,
        )
        # the args names follow the convention of the standalone VAE commandline args
        args_ae = {"init_channels": args.init_channels, "levels": args.enc_levels}
        torch.save({"model": autoencoder.state_dict(), "args": args_ae}, save_dir / "autoencoder")
    return inn


def build_ae(
    args: VaeArgs,
    encoder,
    decoder,
    encoding_size: Optional[EncodingSize],
    feature_group_slices: Optional[Dict[str, List[slice]]],
) -> AutoEncoder:
    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    generator: AutoEncoder
    if args.vae:
        generator = VAE(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            std_transform=args.std_transform,
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
    generator.to(args._device)
    return generator
