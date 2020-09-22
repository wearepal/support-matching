"""Definition of loss and so on."""
from abc import abstractmethod
from typing import Dict, Tuple

import torch
from torch import Tensor, jit
from torch.nn import functional as F

from shared.utils import dot_product, normalized_softmax

from .base import Encoder
from .classifier import Classifier
from .pseudo_labelers import PseudoLabeler

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
        encoder: Encoder,
        classifier: Classifier,
        x: Tensor,
        class_id: Tensor,
        ce_weight: float = 1.0,
        bce_weight: float = 1.0,
    ) -> Tuple[Tensor, LoggingDict]:

        loss = torch.tensor(0.0, device=x.device)
        logging_dict = {}

        if ce_weight or bce_weight:
            z = encoder(x)
            logits = classifier(z)

            if ce_weight:
                ce_loss = (
                    ce_weight * classifier.apply_criterion(logits=logits, targets=class_id).mean()
                )
                loss += ce_loss
                logging_dict["Loss supervised (CE)"] = ce_loss.item()
            if bce_weight:
                preds = F.softmax(logits, dim=-1)
                label = (class_id.unsqueeze(1) == class_id).float()
                mask = torch.ones_like(label)
                bce_loss = bce_weight * _cosine_and_bce(preds, label, mask)
                loss += bce_loss
                logging_dict["Loss supervised (BCE)"] = bce_loss.item()

        return loss, logging_dict

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
def _cosine_and_bce(preds: Tensor, pseudo_label: Tensor, mask: Tensor) -> Tensor:
    """Cosine similarity and then binary cross entropy."""
    # cosine similarity
    cosine_sim = dot_product(preds[:, None, :], preds).clamp(min=0, max=1)
    # binary cross entropy
    unreduced_loss = F.binary_cross_entropy(cosine_sim, pseudo_label, reduction="none")
    return torch.mean(unreduced_loss * mask)
