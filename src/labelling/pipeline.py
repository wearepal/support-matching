from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union, cast

from ranzen import implements
from ranzen.torch.module import DcModule
import torch
from torch import Tensor
import wandb
from wandb.wandb_run import Run

from src.data import DataModule, resolve_device

from .artifact import load_labels_from_artifact, save_labels_as_artifact
from .encoder import ClipVersion, ClipVisualEncoder
from .kmeans import KMeans

__all__ = [
    "LabelFromArtifact",
    "GroundTruthLabeller",
    "KmeansOnClipEncodings",
    "Labeller",
    "NullLabeller",
]


class Labeller(Protocol):
    def run(self, dm: DataModule) -> Tensor | None:
        ...

    def __call__(self, dm: DataModule) -> Tensor | None:
        return self.run(dm=dm)


@dataclass
class KmeansOnClipEncodings(DcModule, Labeller):
    clip_version: ClipVersion = ClipVersion.RN50
    download_root: Optional[str] = None
    ft_steps: int = 1000
    ft_batch_size: int = 16
    ft_lr: float = 1.0e-5
    ft_val_freq: Union[int, float] = 0.1
    ft_val_batches: Union[int, float] = 1.0
    ft_lr: float = 1.0e-5
    enc_batch_size: int = 64

    gpu: int = 0
    spherical: bool = True
    fft_cluster_init: bool = False
    supervised_cluster_init: bool = False
    n_init: int = 10
    save_as_artifact: bool = True

    cache_encoder: bool = False
    encoder: Optional[ClipVisualEncoder] = field(init=False, default=None)
    _fitted_kmeans: Optional[KMeans] = field(init=False, default=None)

    @implements(Labeller)
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
        preds = torch.as_tensor(kmeans.predict(encodings.dep.numpy()), dtype=torch.long)
        if self.save_as_artifact:
            run = cast(Optional[Run], wandb.run)
            save_labels_as_artifact(run=run, labels=preds, datamodule=dm)
        return preds


@dataclass
class LabelFromArtifact(Labeller):
    version: Optional[int] = None  # latest by default
    root: Optional[Path] = None  # artifacts/clustering by default

    @implements(Labeller)
    def run(self, dm: DataModule) -> Tensor:
        return load_labels_from_artifact(
            run=wandb.run, datamodule=dm, version=self.version, root=self.root
        )


@dataclass
class NullLabeller(Labeller):
    @implements(Labeller)
    def run(self, dm: DataModule) -> None:
        return None


@torch.no_grad()
def inject_label_noise(
    labels: Tensor,
    *,
    noise_level: float,
    generator: torch.Generator,
    inplace: bool = True,
) -> Tensor:
    if not 0 <= noise_level <= 1:
        raise ValueError("'noise_level' must be in the range [0, 1].")
    if not inplace:
        labels = labels.clone()
    unique, unique_inv = labels.unique(return_inverse=True)
    num_to_flip = round(noise_level * len(labels))
    to_flip = torch.randperm(len(labels), generator=generator)[:num_to_flip]
    unique_inv[to_flip] += torch.randint(low=1, high=len(unique), size=(num_to_flip,))
    unique_inv[to_flip] %= len(unique)
    return unique[unique_inv]


@dataclass
class GroundTruthLabeller(Labeller):
    label_noise: float = 0
    seed: int = 47
    generator: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not (0 <= self.label_noise <= 1):
            raise ValueError(f"'label_noise' must be in the range [0, 1].")
        self.generator = torch.Generator().manual_seed(self.seed)

    @implements(Labeller)
    def run(self, dm: DataModule) -> Tensor:
        group_ids = dm.group_ids_dep
        # Inject label-noise into the group identifiers.
        if self.label_noise > 0:
            group_ids = inject_label_noise(
                group_ids, noise_level=self.label_noise, generator=self.generator
            )
        return group_ids
