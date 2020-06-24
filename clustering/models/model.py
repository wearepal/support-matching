"""Model that contains all."""
from typing import Tuple, final, Iterator, Optional
from torch import Tensor
import torch
import torch.nn as nn

from .base import Encoder
from .classifier import Classifier
from .pseudo_labelers import PseudoLabeler
from .methods import Method, LoggingDict

__all__ = ["Model", "MultiHeadModel"]


@final
class Model(nn.Module):
    """This class brings everything together into one model object."""

    def __init__(
        self,
        encoder: Encoder,
        classifier: Classifier,
        method: Method,
        pseudo_labeler: PseudoLabeler,
        train_encoder: bool,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.method = method
        self.pseudo_labeler = pseudo_labeler
        self.train_encoder = train_encoder

    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> Tuple[Tensor, LoggingDict]:
        return self.method.supervised_loss(
            encoder=self.encoder,
            classifier=self.classifier,
            x=x,
            class_id=class_id,
            ce_weight=ce_weight,
            bce_weight=bce_weight,
        )

    def unsupervised_loss(self, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.unsupervised_loss(
            encoder=self.encoder,
            pseudo_labeler=self.pseudo_labeler,
            classifier=self.classifier,
            x=x,
        )

    def step(self, grads: Optional[Tensor] = None) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


@final
class MultiHeadModel(nn.Module):
    """This class brings everything together into one model object."""

    def __init__(
        self,
        encoder: Encoder,
        classifiers: nn.ModuleList,
        method: Method,
        pseudo_labeler: PseudoLabeler,
        labeler: Classifier,
        train_encoder: bool,
    ):
        super().__init__()

        self.encoder = encoder
        self.pseudo_labeler = pseudo_labeler
        self.classifiers = classifiers
        self.method = method
        self.labeler = labeler
        self.pseudo_labeler = pseudo_labeler
        self.train_encoder = train_encoder

    def _split_by_label(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        if y is None:
            with torch.no_grad():
                y = self.labeler(x).argmax(dim=-1)
        for i in range(len(self.classifiers)):
            mask = y == i
            if len(mask.nonzero()) > 0:
                yield x[mask], y[mask], mask

    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce: bool = False
    ) -> Tuple[Tensor, LoggingDict]:
        raise RuntimeError("this is broken")
        loss = 0
        for i, (x_i, y_i, _) in enumerate(self._split_by_label(x, class_id=class_id)):
            loss_i, _ = self.method.supervised_loss(
                self.encoder, self.classifiers[i], x_i, y_i, ce=ce
            )
            loss += loss_i
        return loss, {"Loss supervised": loss.item()}

    def unsupervised_loss(self, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        loss = 0
        for i, (x_i, _, _) in enumerate(self._split_by_label(x)):
            loss_i, _ = self.method.unsupervised_loss(
                encoder=self.encoder,
                pseudo_labeler=self.pseudo_labeler,
                classifier=self.classifiers[i],
                x=x_i,
            )
            loss += loss_i
        return loss, {"Loss unsupervised": loss.item()}

    def zero_grad(self) -> None:
        self.classifiers.zero_grad()
        if self.train_encoder:
            self.encoder.zero_grad()

    def step(self, grads: Optional[Tensor] = None) -> None:
        for clf in self.classifiers:
            clf.step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

    def train(self) -> None:
        self.classifiers.train()
        if self.train_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()

    def eval(self) -> None:
        self.encoder.eval()
        self.classifiers.eval()

    def forward(self, x: Tensor) -> Tensor:
        y_dim, s_dim = len(self.classifiers), self.classifiers[0].num_classes
        outputs = x.new_zeros(x.size(0), y_dim, s_dim)
        for i, (x_i, _, mask) in enumerate(self._split_by_label(x)):
            z_i = self.encoder(x_i)
            outputs[mask, i, :] = self.classifiers[i](z_i)
        outputs = outputs.view(x.size(0), -1)

        return outputs
