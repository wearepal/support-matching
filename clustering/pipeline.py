from dataclasses import dataclass, field
from typing import Optional, Union

from ranzen.torch.module import DcModule
import torch
from torch import Tensor

from shared.data import DataModule, resolve_device

from .encoder import ClipVersion, ClipVisualEncoder
from .kmeans import KMeans

__all__ = ["KmeansOnClipEncodings"]


@dataclass
class KmeansOnClipEncodings(DcModule):
    clip_version: ClipVersion = ClipVersion.RN50
    download_root: Optional[str] = None
    ft_steps: int = 1000
    ft_batch_size: int = 16
    ft_lr: float = 1.0e-5
    ft_val_freq: Union[int, float] = 0.1
    ft_val_batches: Union[int, float] = 1
    ft_lr: float = 1.0e-5
    enc_batch_size: int = 64

    gpu: int = 0
    spherical: bool = True
    fft_cluster_init: bool = False
    supervised_cluster_init: bool = False
    n_init: int = 10

    cache_encoder: bool = False
    encoder: Optional[ClipVisualEncoder] = field(init=False, default=None)
    _fitted_kmeans: Optional[KMeans] = field(init=False, default=None)

    def run(self, dm: DataModule, *, use_cached_encoder: bool = False) -> Tensor:
        device = resolve_device(self.gpu)
        kmeans = KMeans(
            spherical=self.spherical,
            supervised_cluster_init=self.supervised_cluster_init,
            n_init=self.n_init,
            device=device,
        )
        if self.encoder is None or not use_cached_encoder:
            encoder = ClipVisualEncoder(
                version=self.clip_version,
                download_root=self.download_root,
            )
            if self.ft_steps > 0:
                encoder.finetune(
                    dm=dm,
                    steps=self.ft_steps,
                    batch_size=self.ft_batch_size,
                    lr=self.ft_lr,
                    val_freq=self.ft_val_freq,
                    val_batches=self.ft_val_batches,
                    device=device,
                )
        else:
            encoder = self.encoder
        encodings = encoder.encode(
            dm=dm,
            batch_size_tr=self.enc_batch_size,
            device=device,
        )
        if self.cache_encoder:
            self.encoder = encoder
        else:
            del encoder
            torch.cuda.empty_cache()

        kmeans.fit(dm=dm, encodings=encodings)
        return torch.as_tensor(kmeans.predict(encodings.dep.numpy()), dtype=torch.long)
