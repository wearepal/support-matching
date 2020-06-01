"""Bundle of all parts."""
from typing import Tuple, final, Optional, Sequence
from torch import Tensor

from .methods import Bundle, Method, LoggingDict

__all__ = ["Model"]


@final
class Model:
    """This class brings everything together into one model object."""

    def __init__(self, bundle: Bundle, method: Method, train_encoder: bool):
        self.bundle = bundle
        self.method = method
        self.train_encoder = train_encoder

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.supervised_loss(self.bundle, x, y)

    def unsupervised_loss(self, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.unsupervised_loss(self.bundle, x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.method.predict(self.bundle, x)

    def step(self, grads: Optional[Sequence[Tensor]] = None) -> None:
        self.bundle.classifier.step(grads)
        if self.train_encoder:
            self.bundle.encoder.step(grads)

    def zero_grad(self) -> None:
        self.bundle.classifier.zero_grad()
        if self.train_encoder:
            self.bundle.encoder.zero_grad()

    def train(self) -> None:
        self.bundle.classifier.train()
        if self.train_encoder:
            self.bundle.encoder.train()
        else:
            self.bundle.encoder.eval()

    def eval(self) -> None:
        self.bundle.encoder.eval()
        self.bundle.classifier.eval()
