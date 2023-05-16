from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

__all__ = ["restore_model", "save_model"]


def save_model(save_dir: Path, *, model: nn.Module, itr: int, best: bool = False) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{itr}.pth"
    save_dict = {
        "model": model.state_dict(),
        "itr": itr,
    }
    torch.save(save_dict, filename)

    return filename


M = TypeVar("M", bound=nn.Module)


def restore_model(filename: Path, *, model: M) -> tuple[M, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(chkpt["model"])
    return model, chkpt["itr"]
