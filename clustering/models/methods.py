"""Definition of loss and so on."""
from abc import abstractmethod
from typing import Tuple, Dict, final

import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F

from shared.utils import dot_product, normalized_softmax

from .base import Encoder
from .labelers import Labeler
from .classifier import Classifier

__all__ = [
    "Bundle",
    "LoggingDict",
    "Method",
    "PseudoLabelEnc",
    "PseudoLabelOutput",
    "PseudoLabelEncNoNorm",
]

LoggingDict = Dict[str, float]


@final
class Bundle(nn.Module):
    """Dumb object that simply holds three other objects."""

    def __init__(self, encoder: Encoder, labeler: Labeler, classifier: Classifier):
        super().__init__()
        self.encoder = encoder
        self.labeler = labeler
        self.classifier = classifier

    def forward(self):
        raise NotImplementedError()


class Method:
    @staticmethod
    def supervised_loss(bundle: Bundle, x: Tensor, y: Tensor) -> Tuple[Tensor, LoggingDict]:
        # default implementation
        z = bundle.encoder(x)
        loss, _ = bundle.classifier.routine(z, y)
        return loss, {"Classification Loss": loss.item()}

    @abstractmethod
    def unsupervised_loss(self, bundle: Bundle, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        """Unsupervised loss."""

    @staticmethod
    def predict(bundle: Bundle, x: Tensor) -> Tensor:
        # default implementation
        return bundle.classifier(bundle.encoder.encode(x))


class PseudoLabelEncNoNorm(Method):
    """Base the pseudo labels on the encodings."""

    @staticmethod
    def unsupervised_loss(bundle: Bundle, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        z = bundle.encoder(x)
        raw_preds = bundle.classifier(z)
        # only do softmax but no real normalization
        preds = F.softmax(raw_preds, dim=-1)
        pseudo_label, mask = bundle.labeler(z)  # base the pseudo labels on the encoding
        loss = _cosine_and_bce(preds, pseudo_label, mask)
        return loss, {"Loss unsupervised": loss.item()}


class PseudoLabelEnc(Method):
    """Base the pseudo labels on the encodings."""

    @staticmethod
    def unsupervised_loss(bundle: Bundle, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        z = bundle.encoder(x)
        raw_preds = bundle.classifier(z)
        # normalize output for cosine similarity
        preds = normalized_softmax(raw_preds)
        pseudo_label, mask = bundle.labeler(z)  # base the pseudo labels on the encoding
        loss = _cosine_and_bce(preds, pseudo_label, mask)
        return loss, {"Loss unsupervised": loss.item()}


class PseudoLabelOutput(Method):
    """Base the pseudo labels on the output of the classifier."""

    @staticmethod
    def unsupervised_loss(bundle: Bundle, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        z = bundle.encoder(x)
        raw_pred = bundle.classifier(z)
        preds = normalized_softmax(raw_pred)
        pseudo_label, mask = bundle.labeler(preds)  # base the pseudo labels on the predictions
        loss = _cosine_and_bce(preds, pseudo_label, mask)
        return loss, {"Loss unsupervised": loss.item()}


@jit.script
def _cosine_and_bce(preds: Tensor, pseudo_label: Tensor, mask: Tensor):
    """Cosine similarity and then binary cross entropy."""
    # cosine similarity
    cosine_sim = dot_product(preds[:, None, :], preds).clamp(min=0, max=1)
    # binary cross entropy
    unreduced_loss = F.binary_cross_entropy(cosine_sim, pseudo_label, reduction="none")
    return torch.mean(unreduced_loss * mask)
