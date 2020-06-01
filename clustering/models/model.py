"""Model that contains all."""
from typing import Tuple, final
from torch import Tensor

from .base import Encoder
from .classifier import Classifier
from .labelers import Labeler
from .methods import Method, LoggingDict

__all__ = ["Model"]


@final
class Model:
    """This class brings everything together into one model object."""

    def __init__(
        self,
        encoder: Encoder,
        labeler: Labeler,
        classifier: Classifier,
        method: Method,
        train_encoder: bool,
    ):
        self.encoder = encoder
        self.labeler = labeler
        self.classifier = classifier
        self.method = method
        self.train_encoder = train_encoder

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.supervised_loss(self.encoder, self.classifier, x, y)

    def unsupervised_loss(self, z: Tensor, raw_preds: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.unsupervised_loss(self.labeler, z=z, raw_preds=raw_preds)

    def __call__(self, x: Tensor) -> Tensor:
        return self.method.predict(self.encoder, self.classifier, x)

    def step(self, grads=None) -> None:
        self.classifier.step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

    def zero_grad(self) -> None:
        self.classifier.zero_grad()
        if self.train_encoder:
            self.encoder.zero_grad()

    def train(self) -> None:
        self.classifier.train()
        if self.train_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()

    def eval(self) -> None:
        self.encoder.eval()
        self.classifier.eval()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def classify(self, z: Tensor) -> Tensor:
        return self.classifier(z)
