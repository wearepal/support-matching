from typing import Dict, List, Optional

from fdm.models import AutoEncoder, EncodingSize, VAE
from shared.configs import Config

__all__ = ["build_ae"]


def build_ae(
    cfg: Config,
    encoder,
    decoder,
    encoding_size: Optional[EncodingSize],
    feature_group_slices: Optional[Dict[str, List[slice]]],
) -> AutoEncoder:
    optimizer_args = {"lr": cfg.fdm.lr, "weight_decay": cfg.fdm.weight_decay}
    model: AutoEncoder
    if cfg.fdm.vae:
        model = VAE(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            vae_std_tform=cfg.fdm.vae_std_tform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        model = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    model.to(cfg.misc._device)
    return model
