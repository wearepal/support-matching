"""Model that contains all."""
from __future__ import annotations
from abc import abstractmethod
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import final

from .base import Encoder
from .classifier import Classifier
from .methods import LoggingDict, Method
from .pseudo_labelers import PseudoLabeler

__all__ = ["Model", "MultiHeadModel"]


class BaseModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        classifier: nn.Module,
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

    @abstractmethod
    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> tuple[Tensor, LoggingDict]:
        """Supervised loss."""

    @abstractmethod
    def unsupervised_loss(self, x: Tensor) -> tuple[Tensor, LoggingDict]:
        """Unsupervised loss."""

    @abstractmethod
    def step(self, grads: Optional[Tensor] = None) -> None:
        """Do one step of gradient descent."""

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

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        pass


@final
class Model(BaseModel):
    """This class brings everything together into one model object."""

    classifier: Classifier

    def __init__(
        self,
        encoder: Encoder,
        classifier: Classifier,
        method: Method,
        pseudo_labeler: PseudoLabeler,
        train_encoder: bool,
    ):
        super().__init__(
            encoder=encoder,
            classifier=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=train_encoder,
        )

    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> tuple[Tensor, LoggingDict]:
        return self.method.supervised_loss(
            encoder=self.encoder,
            classifier=self.classifier,
            x=x,
            class_id=class_id,
            ce_weight=ce_weight,
            bce_weight=bce_weight,
        )

    def unsupervised_loss(self, x: Tensor) -> tuple[Tensor, LoggingDict]:
        z = self.encoder(x)
        raw_preds = self.classifier(z)
        return self.method.unsupervised_loss(
            pseudo_labeler=self.pseudo_labeler, z=z, raw_preds=raw_preds
        )

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.classifier.step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

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
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        if y is None:
            with torch.no_grad():
                y = self.labeler(x).argmax(dim=-1)
        for i in range(len(self.classifiers)):
            mask = y == i
            if len(mask.nonzero()) > 0:
                yield x[mask], y[mask], mask

    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce: bool = False
    ) -> tuple[Tensor, LoggingDict]:
        raise RuntimeError("this is broken")
        loss = 0
        for i, (x_i, y_i, _) in enumerate(self._split_by_label(x, class_id=class_id)):
            loss_i, _ = self.method.supervised_loss(
                self.encoder, self.classifiers[i], x_i, y_i, ce=ce
            )
            loss += loss_i
        return loss, {"Loss supervised": loss.item()}

    def unsupervised_loss(self, x: Tensor) -> tuple[Tensor, LoggingDict]:
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
