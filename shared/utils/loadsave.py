"""Functions related to saving and loading results."""
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from shared.configs import BaseArgs

__all__ = ["save_results", "load_results"]


def save_results(
    cluster_ids: torch.Tensor, class_ids: torch.Tensor, save_path: Path, flags: Dict[str, Any]
) -> Path:
    """Save a tensor in a file."""
    save_dict = {
        "args": flags,
        "cluster_ids": cluster_ids,
        "class_ids": class_ids,
    }
    torch.save(save_dict, save_path)
    print(
        f"To make use of the generated cluster labels:\n--cluster-label-file {save_path.resolve()}"
    )
    return save_path


def load_results(
    args: BaseArgs, check: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load a tensor from a file."""
    data = torch.load(args.cluster_label_file, map_location=torch.device("cpu"))
    if check:
        saved_args = data["args"]
        assert saved_args["dataset"] == args.dataset, f'{saved_args["dataset"]} != {args.dataset}'
        assert (
            saved_args["data_pcnt"] == args.data_pcnt
        ), f'{saved_args["data_pcnt"]} != {args.data_pcnt}'
        assert (
            saved_args["data_split_seed"] == args.data_split_seed
        ), f'{saved_args["data_split_seed"]} != {args.data_split_seed}'
        assert (
            saved_args["context_pcnt"] == args.context_pcnt
        ), f'{saved_args["context_pcnt"]} != {args.context_pcnt}'
        assert (
            saved_args["test_pcnt"] == args.test_pcnt
        ), f'{saved_args["test_pcnt"]} != {args.test_pcnt}'
    class_ids = data["class_ids"] if "class_ids" in data else torch.zeros_like(data["cluster_ids"])
    return data["cluster_ids"], class_ids, data["args"]
