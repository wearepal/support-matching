"""Definition of loss and so on."""
from abc import abstractmethod
from typing import Tuple, Dict

import torch
from torch import Tensor, jit
from torch.nn import functional as F

from shared.utils import dot_product, normalized_softmax

from .base import Encoder
from .pseudo_labelers import PseudoLabeler
from .classifier import Classifier

__all__ = [
    "LoggingDict",
    "Method",
    "PseudoLabelEnc",
    "PseudoLabelOutput",
    "PseudoLabelEncNoNorm",
]

LoggingDict = Dict[str, float]


class Method:
    @staticmethod
    def supervised_loss(
        encoder: Encoder, classifier: Classifier, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, LoggingDict]:
        # default implementation
        z = encoder(x)
        loss, _ = classifier.routine(z, y)
        return loss, {"Classification Loss": loss.item()}

    @abstractmethod
    def unsupervised_loss(
        self, encoder: Encoder, pseudo_labeler: PseudoLabeler, classifier: Classifier, x: Tensor
    ) -> Tuple[Tensor, LoggingDict]:
        """Unsupervised loss."""

    @staticmethod
    def predict(encoder: Encoder, classifier: Classifier, x: Tensor) -> Tensor:
        # default implementation
        return classifier(encoder.encode(x))


class PseudoLabelEncNoNorm(Method):
    """Base the pseudo labels on the encodings."""

    @staticmethod
    def unsupervised_loss(
        encoder: Encoder, pseudo_labeler: PseudoLabeler, classifier: Classifier, x: Tensor
    ) -> Tuple[Tensor, LoggingDict]:
        z = encoder(x)
        raw_preds = classifier(z)
        # only do softmax but no real normalization
        preds = F.softmax(raw_preds, dim=-1)
        pseudo_label, mask = pseudo_labeler(z)  # base the pseudo labels on the encoding
        loss = _cosine_and_bce(preds, pseudo_label, mask)
        return loss, {"Loss unsupervised": loss.item()}


class PseudoLabelEnc(Method):
    """Base the pseudo labels on the encodings."""

    @staticmethod
    def unsupervised_loss(
        encoder: Encoder, pseudo_labeler: PseudoLabeler, classifier: Classifier, x: Tensor
    ) -> Tuple[Tensor, LoggingDict]:
        z = encoder(x)
        raw_preds = classifier(z)
        # normalize output for cosine similarity
        preds = normalized_softmax(raw_preds)
        pseudo_label, mask = pseudo_labeler(z)  # base the pseudo labels on the encoding
        loss = _cosine_and_bce(preds, pseudo_label, mask)
        return loss, {"Loss unsupervised": loss.item()}


class PseudoLabelOutput(Method):
    """Base the pseudo labels on the output of the classifier."""

    @staticmethod
    def unsupervised_loss(
        encoder: Encoder, pseudo_labeler: PseudoLabeler, classifier: Classifier, x: Tensor
    ) -> Tuple[Tensor, LoggingDict]:
        z = encoder(x)
        raw_pred = classifier(z)
        preds = normalized_softmax(raw_pred)
        pseudo_label, mask = pseudo_labeler(preds)  # base the pseudo labels on the predictions
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
