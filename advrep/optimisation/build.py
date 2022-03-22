from __future__ import annotations

from advrep.models import AutoEncoder, EncodingSize, VAE
from shared.configs import Config

__all__ = ["build_ae"]


def build_ae(
    cfg: Config,
    encoder,
    decoder,
    encoding_size: EncodingSize | None,
    s_dim: int,
    feature_group_slices: dict[str, list[slice]] | None,
) -> AutoEncoder:
    optimizer_args = {"lr": cfg.adapt.lr, "weight_decay": cfg.adapt.weight_decay}
    model: AutoEncoder
    if cfg.adapt.vae:
        model = VAE(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            s_dim=s_dim,
            vae_std_tform=cfg.adapt.vae_std_tform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        model = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            s_dim=s_dim,
            zs_transform=cfg.adapt.zs_transform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    model.to(cfg.train.device)
    return model
