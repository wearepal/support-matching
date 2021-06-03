from __future__ import annotations
from collections.abc import Iterable, Iterator
from datetime import datetime
import random
import time
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn

__all__ = ["count_parameters", "random_seed", "inf_generator", "Event"]


def count_parameters(model: nn.Module) -> int:
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_seed(seed_value: int, use_cuda: bool) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


T = TypeVar("T")


def inf_generator(iterable: Iterable[T]) -> Iterator[T]:
    """Get DataLoaders in a single infinite loop.

    for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iter(iterable)
    # try to take one element to ensure that the iterator is not empty
    first_value = next(iterator, None)
    if first_value is not None:
        yield first_value
    else:
        raise RuntimeError("The given iterable is empty.")
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Event:
    """Emulates torch.cuda.Event, but supports running on a CPU too.

    Examples:
    >>> from kit.torch import Event
    >>> with Event() as event:
    >>>     y = some_nn_module(x)
    >>> print(event.time)
    """

    def __init__(self) -> None:
        self.time = 0.0
        self._cuda = torch.cuda.is_available()  # type: ignore
        self._event_start: torch.cuda.Event | datetime

    def __enter__(self) -> Event:
        """Mark a time.

        Mimics torch.cuda.Event.
        """
        if self._cuda:
            self._event_start = torch.cuda.Event(enable_timing=True)
            self._event_start.record()
        else:
            self._event_start = datetime.now()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._cuda:
            event_end = torch.cuda.Event(enable_timing=True)
            event_end.record()
            torch.cuda.synchronize()
            assert isinstance(self._event_start, torch.cuda.Event)
            self.time = self._event_start.elapsed_time(event_end)
        else:
            assert isinstance(self._event_start, datetime)
            self.time = datetime.now().microsecond - self._event_start.microsecond

    def __repr__(self) -> str:
        return f"Event of duration: {self.time}"
