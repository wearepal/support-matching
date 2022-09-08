from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict
from typing_extensions import Self

from conduit.data.datasets import ImageTform
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from loguru import logger
import numpy as np
import numpy.typing as npt
from ranzen import gcopy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from src.data import DataModule, labels_to_group_id

__all__ = [
    "Encodings",
    "generate_encodings",
]


@dataclass
class Encodings:
    """Result of encoding the data."""

    train: Tensor  # This is used to compute the pre-defined starting points.
    train_labels: Tensor  # Same as above.
    dep: Tensor  # This is used to compute the other starting points.
    test: npt.NDArray  # This is used for evaluation.
    test_labels: npt.NDArray[np.int32]  # Same as above.

    def normalize_(self, p: float = 2) -> None:
        self.train = F.normalize(self.train, dim=1, p=2)
        self.dep = F.normalize(self.dep, dim=1, p=2)
        self.test = F.normalize(torch.as_tensor(self.test), dim=1, p=2).numpy()

    def save(self, fpath: Path | str) -> None:
        fpath = Path(fpath)
        logger.info(f"Saving encodings to '{fpath.resolve()}'")
        np.savez_compressed(file=Path(fpath), **asdict(self))
        logger.info("Done.")

    @property
    def to_cluster(self) -> npt.NDArray:
        return torch.cat([self.train, self.dep], dim=0).numpy()

    @classmethod
    def from_npz(cls: type[Self], fpath: Path | str) -> Self:
        logger.info("Loading encodings from file...")
        with Path(fpath).open("rb") as f:
            loaded: NpzContent = np.load(f)
            enc = cls(
                train=torch.from_numpy(loaded["train"]),
                train_labels=torch.from_numpy(loaded["train_ids"]),
                dep=torch.from_numpy(loaded["dep"]),
                test=loaded["test"],
                test_labels=loaded["test_ids"],
            )
        return enc


@torch.no_grad()
def generate_encodings(
    dm: DataModule,
    *,
    encoder: nn.Module,
    device: str | torch.device,
    batch_size_tr: int | None = None,
    batch_size_te: int | None = None,
    transforms: ImageTform | None = None,
    save_path: Path | str | None = None,
) -> Encodings:
    """Generate encodings by putting the data through a pre-trained model."""
    dm = gcopy(dm, deep=False)
    if transforms is not None:
        dm.set_transforms_all(transforms)

    encoder.to(device)

    train_enc, train_group_ids = encode_with_group_ids(
        encoder,
        dl=dm.train_dataloader(eval=True, batch_size=batch_size_tr),
        device=device,
    )
    deployment_enc, _ = encode_with_group_ids(
        encoder,
        dl=dm.deployment_dataloader(eval=True, batch_size=batch_size_te),
        device=device,
    )
    test_enc, test_group_ids = encode_with_group_ids(
        encoder, dl=dm.test_dataloader(), device=device
    )
    torch.cuda.empty_cache()

    encodings = Encodings(
        train=train_enc,
        train_labels=train_group_ids,
        dep=deployment_enc,
        test=test_enc.numpy(),
        test_labels=test_group_ids.numpy(),
    )
    if save_path is not None:
        encodings.save(save_path)
    return encodings


@torch.no_grad()
def encode_with_group_ids(
    model: nn.Module,
    *,
    dl: CdtDataLoader[TernarySample[Tensor]],
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    encoded: list[Tensor] = []
    group_ids: list[Tensor] = []
    with torch.no_grad():
        for sample in tqdm(dl, total=len(dl), desc="Encoding dataset"):
            enc = model(sample.x.to(device, non_blocking=True)).detach()
            # normalize so we're doing cosine similarity
            encoded.append(enc.cpu())
            group_ids.append(labels_to_group_id(s=sample.s, y=sample.y, s_count=2))
    logger.info("Done.")
    return torch.cat(encoded, dim=0), torch.cat(group_ids, dim=0)


class NpzContent(TypedDict):
    """Content of the npz file (which is basically a dictionary)."""

    train: npt.NDArray
    train_ids: npt.NDArray
    dep: npt.NDArray
    test: npt.NDArray
    test_ids: npt.NDArray[np.int32]
