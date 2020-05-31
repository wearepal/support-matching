"""Functions related to saving and loading results."""
from pathlib import Path

import torch

from shared.configs import BaseArgs

__all__ = ["save_results", "load_results"]


def save_results(args: BaseArgs, cluster_ids: torch.Tensor, results_dir: Path) -> Path:
    """Save a tensor in a file."""
    save_path = results_dir / "cluster_results.pth"
    save_dict = {
        "args": args.as_dict(),
        "cluster_ids": cluster_ids,
    }
    torch.save(save_dict, save_path)
    print(f"Saved results in {save_path.resolve()}")
    return save_path


def load_results(args: BaseArgs, cluster_label_file: Path) -> torch.Tensor:
    """Load a tensor from a file."""
    data = torch.load(cluster_label_file, map_location=torch.device("cpu"))
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
    args.cluster_label_file = str(cluster_label_file)
    return data["cluster_ids"]
