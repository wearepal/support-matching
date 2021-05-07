from __future__ import annotations

from shared.configs import Config
from suds.models import AutoEncoder, EncodingSize, Vae

__all__ = ["build_ae"]


def build_ae(
    cfg: Config,
    encoder,
    decoder,
    encoding_size: EncodingSize | None,
    feature_group_slices: dict[str, list[slice]] | None,
) -> AutoEncoder:
    optimizer_args = {"lr": cfg.adv.lr, "weight_decay": cfg.adv.weight_decay}
    model: AutoEncoder
    if cfg.adv.vae:
        model = Vae(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            vae_std_tform=cfg.adv.vae_std_tform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        model = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            zs_transform=cfg.enc.zs_transform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    model.to(cfg.misc.device)
    return model