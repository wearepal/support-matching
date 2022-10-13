from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional, cast

import clip
from loguru import logger
from ranzen.misc import gcopy
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor
import torch.nn as nn

from src.data import DataModule

from .encode import Encodings, generate_encodings
from .finetuning import FineTuner

__all__ = ["ClipVersion", "ClipVisualEncoder"]


class ClipVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B16 = "ViT-B/16"
    ViT_B32 = "ViT-B/32"
    ViT_L14 = "ViT-L/14"


class ClipVisualEncoder(nn.Module):
    def __init__(
        self, version: ClipVersion = ClipVersion.RN50, *, download_root: Optional[str] = None
    ) -> None:
        super().__init__()
        logger.info("Loading CLIP model (downloading if needed)...")
        model, self.transforms = clip.load(
            name=version.value, device="cpu", download_root=download_root  # type: ignore
        )
        logger.info("Done.")
        self.encoder = model.visual
        self.out_dim = cast(int, self.encoder.output_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.encoder(x)

    @torch.no_grad()
    def load_from_path(self, fpath: Path | str) -> None:
        fpath = Path(fpath)
        if fpath.exists():
            logger.info(f"Loading model weights from '{fpath.resolve()}'")
            self.load_state_dict(torch.load(fpath))
            logger.info("Done.")
        else:
            raise RuntimeError(f"Checkpoint {fpath.resolve()} does not exist.")

    @torch.no_grad()
    def encode(
        self,
        dm: DataModule,
        *,
        device: str | torch.device,
        batch_size_tr: int,
        batch_size_te: int | None = None,
    ) -> Encodings:
        return generate_encodings(
            dm=dm,
            encoder=self,
            transforms=self.transforms,
            batch_size_tr=batch_size_tr,
            batch_size_te=batch_size_te,
            device=device,
        )

    def finetune(
        self,
        dm: DataModule,
        *,
        batch_size: int = 16,
        steps: int,
        val_freq: int | float = 0.1,
        lr: float = 1.0e-5,
        device: str | torch.device | int = 0,
        val_batches: int | float = 1.0,
    ) -> None:
        dm = gcopy(dm, deep=False)
        dm.set_transforms_all(self.transforms)
        finetuner = FineTuner(
            batch_size=batch_size,
            steps=steps,
            lr=lr,
            val_freq=val_freq,
            val_batches=val_batches,
            loss_fn=CrossEntropyLoss(reduction="mean"),
            device=device,
        )
        logger.info(f"Fine-tuning visual encoder for {steps} steps with batch size {batch_size}.")
        finetuner.run(dm=dm, backbone=self, out_dim=self.out_dim)
