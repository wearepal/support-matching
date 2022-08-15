from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, TypedDict
from typing_extensions import Self

import clip
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
import numpy as np
import numpy.typing as npt
from ranzen import gcopy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from shared.data.data_module import DataModule
from shared.data.utils import labels_to_group_id


class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


@dataclass
class Encodings:
    """Result of encoding the data."""

    to_cluster: npt.NDArray  # This is passed directly to sklearn's kmeans algorithm.
    train: Tensor  # This is used to compute the pre-defined starting points.
    train_labels: Tensor  # Same as above.
    dep: Tensor  # This is used to compute the other starting points.
    test: npt.NDArray  # This is used for evaluation.
    test_labels: npt.NDArray[np.int32]  # Same as above.

    def save(self, fpath: Path | str) -> None:
        fpath = Path(fpath)
        print(f"Saving encodings to '{fpath.resolve()}'")
        np.savez_compressed(file=Path(fpath), **asdict(self))
        print("Done.")

    @classmethod
    def from_npz(cls: type[Self], fpath: Path | str) -> Self:
        print("Loading encodings from file...")
        with Path(fpath).open("rb") as f:
            loaded: NpzContent = np.load(f)
            enc = cls(
                train=torch.from_numpy(loaded["train"]),
                train_labels=torch.from_numpy(loaded["train_ids"]),
                dep=torch.from_numpy(loaded["dep"]),
                test=loaded["test"],
                test_labels=loaded["test_ids"],
                to_cluster=np.concatenate([loaded["train"], loaded["dep"]], axis=0),
            )
        return enc


def generate_encodings(
    dm: DataModule,
    clip_version: CLIPVersion = CLIPVersion.RN50,
    download_root: str | None = None,
    batch_size_tr: int | None = None,
    batch_size_te: int | None = None,
    model_path: Path | str = "./finetuned.pt",
) -> Encodings:
    """Generate encodings by putting the data through a pre-trained model."""
    dm = gcopy(dm, deep=False)
    print("Loading CLIP model (downloading if needed)...", flush=True)
    model, transforms = clip.load(
        name=clip_version.value, device="cpu", download_root=download_root  # type: ignore
    )
    dm.set_transforms_all(transforms)

    print("Done.")
    visual_model = model.visual

    model_path = Path(model_path)
    if model_path.exists():
        print("Loading finetuned weights...")
        visual_model.load_state_dict(torch.load(model_path))
        print("Done.")
    visual_model.to("cuda:0")

    train_enc, train_group_ids = encode(
        visual_model, dl=dm.train_dataloader(eval=True, batch_size=batch_size_tr)
    )
    deployment_enc, _ = encode(
        visual_model, dl=dm.deployment_dataloader(eval=True, batch_size=batch_size_te)
    )
    test_enc, test_group_ids = encode(visual_model, dl=dm.test_dataloader())
    del model
    del visual_model
    torch.cuda.empty_cache()

    to_save: NpzContent = {
        "train": train_enc.numpy(),
        "train_ids": train_group_ids.numpy(),
        "dep": deployment_enc.numpy(),
        "test": test_enc.numpy(),
        "test_ids": test_group_ids.numpy(),
    }
    save_encoding(to_save)
    return Encodings(
        to_cluster=torch.cat([train_enc, deployment_enc], dim=0).numpy(),
        train=train_enc,
        train_labels=train_group_ids,
        dep=deployment_enc,
        test=test_enc.numpy(),
        test_labels=test_group_ids.numpy(),
    )


def encode(model: nn.Module, *, dl: CdtDataLoader[TernarySample[Tensor]]) -> Tuple[Tensor, Tensor]:
    encoded: List[Tensor] = []
    group_ids: List[Tensor] = []
    print("Beginning encoding...", flush=True)
    with torch.no_grad():
        for sample in tqdm(dl, total=len(dl)):
            enc = model(sample.x.to("cuda:0", non_blocking=True)).detach()
            # normalize so we're doing cosine similarity
            encoded.append(F.normalize(enc, dim=1, p=2).cpu())
            group_ids.append(labels_to_group_id(s=sample.s, y=sample.y, s_count=2))
    print("done.")
    return torch.cat(encoded, dim=0), torch.cat(group_ids, dim=0)


class NpzContent(TypedDict):
    """Content of the npz file (which is basically a dictionary)."""

    train: npt.NDArray
    train_ids: npt.NDArray
    dep: npt.NDArray
    test: npt.NDArray
    test_ids: npt.NDArray[np.int32]


def save_encoding(all_encodings: NpzContent, file: Path | str = "encoded_celeba.npz") -> None:
    print("Saving encodings to 'encoded_celeba.npz'...")
    np.savez_compressed(file=Path(file), **all_encodings)
    print("Done.")
