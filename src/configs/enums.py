from enum import Enum, auto

__all__ = [
    "EvalTrainData",
    "FsMethod",
]


class EvalTrainData(Enum):
    """Dataset to use for training during evaluation."""

    train = auto()
    deployment = auto()


class FsMethod(Enum):
    erm = auto()
    dro = auto()
    gdro = auto()
    lff = auto()
