"""Functions related to saving and loading results."""
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from torch import Tensor

__all__ = [
    "ClusterResults",
]


class ClusterResults(NamedTuple):
    """Information that the fcm code passes on to advrep."""

    flags: Dict[str, Any]
    cluster_ids: Tensor
    class_ids: Tensor
    enc_path: Path
    context_metrics: Optional[Dict[str, float]]
    test_metrics: Optional[Dict[str, float]] = None
