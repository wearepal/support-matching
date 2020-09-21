"""Functions related to saving and loading results."""
from pathlib import Path
from typing import Any, Dict, NamedTuple

import torch

from shared.configs import BaseArgs

__all__ = ["ClusterResults", "save_results", "load_results"]


class ClusterResults(NamedTuple):
    """Information that the fcm code passes on to fdm."""

    flags: Dict[str, Any]
    cluster_ids: torch.Tensor
    class_ids: torch.Tensor
    enc_path: Path
    context_acc: float
    test_acc: float = float("nan")


def save_results(save_path: Path, cluster_results: ClusterResults) -> Path:
    """Save a tensor in a file."""
    save_dict = {
        "args": cluster_results.flags,
        "cluster_ids": cluster_results.cluster_ids,
        "class_ids": cluster_results.class_ids,
        "enc_path": str(cluster_results.enc_path.resolve()),
        "test_acc": cluster_results.test_acc,
        "context_acc": cluster_results.context_acc,
    }
    torch.save(save_dict, save_path)
    print(
        f"To make use of the generated cluster labels:\n--cluster-label-file {save_path.resolve()}"
    )
    return save_path


def load_results(args: BaseArgs, check: bool = True) -> ClusterResults:
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
    return ClusterResults(
        flags=data["args"],
        cluster_ids=data["cluster_ids"],
        class_ids=class_ids,
        enc_path=Path(data.get("enc_path", "")),
        context_acc=data.get("context_acc", float("nan")),
        test_acc=data.get("test_acc", float("nan")),
    )
