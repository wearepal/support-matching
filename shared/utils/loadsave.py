"""Functions related to saving and loading results."""
import logging
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from torch import Tensor

__all__ = [
    "ClusterResults",
]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class ClusterResults(NamedTuple):
    """Information that the fcm code passes on to advrep."""

    flags: Dict[str, Any]
    cluster_ids: Tensor
    class_ids: Tensor
    enc_path: Path
    context_metrics: Optional[Dict[str, float]]
    test_metrics: Optional[Dict[str, float]] = None
