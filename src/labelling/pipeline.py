from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast
from typing_extensions import override

import conduit.metrics as cdtm
from conduit.models.utils import prefix_keys
from loguru import logger
from ranzen.torch.module import DcModule
import torch
from torch import Tensor
import wandb
from wandb.wandb_run import Run

from src.data import DataModule, resolve_device
from src.evaluation.metrics import print_metrics
from src.models import Classifier, OptimizerCfg
from src.utils import to_item

from .artifact import load_labels_from_artifact, save_labels_as_artifact
from .encode import Encodings, encode_with_group_ids
from .encoder import ClipVersion, ClipVisualEncoder
from .finetuning import FineTuneParams
from .kmeans import KMeans
from .noise import (
    ClnMetric,
    centroidal_label_noise,
    sample_noise_indices,
    uniform_label_noise,
)

__all__ = [
    "CentroidalLabelNoiser",
    "ClipClassifier",
    "GroundTruthLabeller",
    "KmeansOnClipEncodings",
    "LabelFromArtifact",
    "Labeller",
    "NullLabeller",
    "UniformLabelNoiser",
    "centroidal_label_noise",
    "uniform_label_noise",
]


class Labeller(ABC):
    @abstractmethod
    def run(self, dm: DataModule) -> Optional[Tensor]:
        raise NotImplementedError()

    def __call__(self, dm: DataModule) -> Optional[Tensor]:
        return self.run(dm=dm)


@dataclass(repr=False, eq=False)
class KmeansOnClipEncodings(DcModule, Labeller):
    clip_version: ClipVersion = ClipVersion.RN50
    download_root: Optional[str] = None
    ft: FineTuneParams = field(default_factory=lambda: FineTuneParams(steps=1000))
    enc_batch_size: int = 64

    gpu: int = 0
    spherical: bool = True
    fft_cluster_init: bool = False
    supervised_cluster_init: bool = False
    n_init: int = 10
    save_as_artifact: bool = True
    artifact_name: Optional[str] = None

    cache_encoder: bool = False
    encodings_path: Optional[Path] = None

    encoder: Optional[ClipVisualEncoder] = field(
        init=False, default=None, metadata={"omegaconf_ignore": True}
    )
    _fitted_kmeans: Optional[KMeans] = field(
        init=False, default=None, metadata={"omegaconf_ignore": True}
    )

    @override
    def run(self, dm: DataModule, *, use_cached_encoder: bool = False) -> Tensor:
        device = resolve_device(self.gpu)
        if self.encodings_path is not None and self.encodings_path.exists():
            encodings = Encodings.from_npz(self.encodings_path)
        else:
            if self.encoder is None or not use_cached_encoder:
                encoder = ClipVisualEncoder(
                    version=self.clip_version, download_root=self.download_root
                )
                if self.ft.steps > 0:
                    encoder.finetune(dm=dm, params=self.ft, device=device)
            else:
                encoder = self.encoder
            encodings = encoder.encode(dm=dm, batch_size_tr=self.enc_batch_size, device=device)
            if self.encodings_path is not None:
                encodings.save(self.encodings_path)
            if self.cache_encoder:
                self.encoder = encoder
            else:
                del encoder
                torch.cuda.empty_cache()

        kmeans = KMeans(
            spherical=self.spherical,
            supervised_cluster_init=self.supervised_cluster_init,
            n_init=self.n_init,
            device=device,
        )
        kmeans.fit(dm=dm, encodings=encodings)
        self._fitted_kmeans = kmeans
        preds = torch.as_tensor(kmeans.predict(encodings.dep.numpy()), dtype=torch.long)
        if self.save_as_artifact:
            run = cast(Optional[Run], wandb.run)
            save_labels_as_artifact(
                run=run, labels=preds, datamodule=dm, artifact_name=self.artifact_name
            )
        return preds


@dataclass(eq=False)
class ClipClassifier(Labeller):
    clip_version: ClipVersion = ClipVersion.RN50
    download_root: Optional[str] = None
    ft: FineTuneParams = field(default_factory=lambda: FineTuneParams(steps=1000))
    batch_size_te: int = 64

    gpu: int = 0
    save_as_artifact: bool = True
    artifact_name: Optional[str] = None

    cache_encoder: bool = False

    @torch.no_grad()
    def evaluate(
        self, g_pred: Tensor, *, g_true: Tensor, use_wandb: bool, prefix: Optional[str] = None
    ) -> dict[str, float]:
        metrics = {
            "Accuracy": to_item(cdtm.accuracy(y_pred=g_pred, y_true=g_true)),
            "Balanced_Accuracy": to_item(
                cdtm.subclass_balanced_accuracy(y_pred=g_pred, y_true=g_true, s=g_true)
            ),
            "Robust_Accuracy": to_item(
                cdtm.robust_accuracy(y_pred=g_pred, y_true=g_true, s=g_true)
            ),
        }
        if prefix is not None:
            metrics = prefix_keys(metrics, prefix=prefix, sep="/")
        if use_wandb:
            wandb.log(metrics)

        return metrics

    @override
    def run(self, dm: DataModule, *, use_cached_encoder: bool = False) -> Tensor:
        device = resolve_device(self.gpu)
        encoder = ClipVisualEncoder(version=self.clip_version, download_root=self.download_root)
        ft_model = encoder.finetune(dm=dm, params=self.ft, device=device)
        classifier = Classifier(model=ft_model, opt=OptimizerCfg())
        preds = classifier.predict(
            dm.deployment_dataloader(eval=True, batch_size=self.batch_size_te),
            device=device,
            with_soft=True,
        )
        g_pred = preds.y_true
        g_true = preds.group_ids
        metrics = self.evaluate(g_pred=g_pred, g_true=g_true, prefix="labelling", use_wandb=True)
        print_metrics(metrics)

        if self.save_as_artifact:
            run = cast(Optional[Run], wandb.run)
            save_labels_as_artifact(
                run=run, labels=g_pred, datamodule=dm, artifact_name=self.artifact_name
            )
        return g_pred


@dataclass(eq=False)
class LabelFromArtifact(Labeller):
    version: Optional[int] = None  # latest by default
    artifact_name: Optional[str] = None
    root: Optional[Path] = None  # artifacts/clustering by default

    @override
    def run(self, dm: DataModule) -> Tensor:
        return load_labels_from_artifact(
            run=wandb.run,
            datamodule=dm,
            version=self.version,
            root=self.root,
            name=self.artifact_name,
        )


@dataclass(eq=False)
class NullLabeller(Labeller):
    @override
    def run(self, dm: DataModule) -> None:
        return None


@dataclass(eq=False)
class GroundTruthLabeller(Labeller):
    seed: int = 47

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)

    @override
    def run(self, dm: DataModule) -> Tensor:
        return dm.group_ids_dep


@dataclass(eq=False)
class LabelNoiser(Labeller):
    level: float = 0.10
    seed: int = 47
    weighted_index_sampling: bool = True

    def __post_init__(self) -> None:
        if not (0 <= self.level <= 1):
            raise ValueError("'label_noise' must be in the range [0, 1].")

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)

    @abstractmethod
    def _noise(self, dep_ids: Tensor, *, flip_inds: Tensor, dm: DataModule) -> Tensor:
        raise NotImplementedError()

    @override
    def run(self, dm: DataModule) -> Tensor:
        group_ids = dm.group_ids_dep
        logger.info(
            f"Injecting noise into ground-truth labels with noise level '{self.level}'"
            f" ({self.level * 100}% of samples will have their labels altered)."
        )
        if self.level > 0:  # At level 0, the original labels are preserved in their entirety
            flip_inds = sample_noise_indices(
                labels=group_ids,
                level=self.level,
                generator=self.generator,
                weighted=self.weighted_index_sampling,
            )
            # Inject label-noise into the group identifiers.
            group_ids = self._noise(dep_ids=group_ids, flip_inds=flip_inds, dm=dm)
        return group_ids


@dataclass(eq=False)
class UniformLabelNoiser(LabelNoiser):
    @override
    def _noise(self, dep_ids: Tensor, *, flip_inds: Tensor, dm: DataModule) -> Tensor:
        return uniform_label_noise(
            labels=dep_ids, indices=flip_inds, generator=self.generator, inplace=True
        )


@dataclass(eq=False)
class CentroidalLabelNoiser(LabelNoiser):
    metric: ClnMetric = ClnMetric.COSINE
    clip_version: ClipVersion = ClipVersion.RN50
    download_root: Optional[str] = None
    enc_batch_size: int = 64
    gpu: int = 0

    @override
    def _noise(self, dep_ids: Tensor, *, flip_inds: Tensor, dm: DataModule) -> Tensor:
        device = resolve_device(self.gpu)
        encoder = ClipVisualEncoder(version=self.clip_version, download_root=self.download_root)
        encodings, _ = encode_with_group_ids(
            model=encoder,
            dl=dm.deployment_dataloader(eval=True, batch_size=self.enc_batch_size),
            device=device,
        )
        return centroidal_label_noise(
            labels=dep_ids,
            indices=flip_inds,
            encodings=encodings,
            generator=self.generator,
            inplace=True,
        )
