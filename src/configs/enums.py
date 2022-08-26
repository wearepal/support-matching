from enum import Enum, auto

__all__ = [
    "AggregatorType",
    "ClusteringLabel",
    "DiscriminatorLoss",
    "DiscriminatorMethod",
    "EncoderType",
    "EvalTrainData",
    "FsMethod",
    "MMDKernel",
]


class ClusteringLabel(Enum):
    """Which attribute(s) to cluster on."""

    s = auto()
    y = auto()
    both = auto()
    manual = auto()


class EncoderType(Enum):
    """Encoder type."""

    ae = auto()
    vae = auto()
    rotnet = auto()


class DiscriminatorMethod(Enum):
    """Method of distribution-discrimination."""

    nn = auto()
    mmd = auto()


class AggregatorType(Enum):
    """Which aggregation function to use."""

    from src.arch.aggregation import GatedAggregator, KvqAggregator

    kvq = KvqAggregator
    gated = GatedAggregator


class DiscriminatorLoss(Enum):
    """Which type of adversarial loss to use."""

    wasserstein = auto()
    logistic_ns = auto()
    logistic = auto()


class EvalTrainData(Enum):
    """Dataset to use for training during evaluation."""

    train = auto()
    deployment = auto()


class FsMethod(Enum):
    erm = auto()
    dro = auto()
    gdro = auto()
    lff = auto()
