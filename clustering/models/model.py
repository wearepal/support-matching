"""Model that contains all."""
from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from typing import Optional, cast

from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from clustering.optimisation.utils import get_class_id
from shared.configs.enums import ClusteringLabel

from .base import Encoder
from .classifier import Classifier
from .methods import LoggingDict, Method
from .pseudo_labelers import PseudoLabeler

__all__ = ["BaseModel", "FlatModel", "HierarchicalModel"]


class BaseModel(nn.Module):
    """This class brings everything together into one model object."""

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
        self, x: Tensor, s: Tensor, y: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
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


class FlatModel(BaseModel):
    """Predict clusters in a flat manner."""

    classifier: Classifier

    def __init__(
        self,
        encoder: Encoder,
        classifier: Classifier,
        method: Method,
        pseudo_labeler: PseudoLabeler,
        train_encoder: bool,
        to_cluster: ClusteringLabel,
        s_count: int,
    ):
        super().__init__(
            encoder=encoder,
            classifier=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=train_encoder,
        )
        self.to_cluster = to_cluster
        self.s_count = s_count

    def supervised_loss(
        self, x: Tensor, s: Tensor, y: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> tuple[Tensor, LoggingDict]:
        class_id = get_class_id(s=s, y=y, s_count=self.s_count, to_cluster=self.to_cluster)
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
        preds = F.softmax(raw_preds, dim=-1)
        return self.method.unsupervised_loss(
            pseudo_labeler=self.pseudo_labeler, z=z, preds=preds
        )

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.classifier.step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

    def forward(self, x: Tensor) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        return self.classifier(self.encoder(x)), None


class HierarchicalModel(BaseModel):
    """Using separate cluster heads for s and y."""

    classifier: nn.ModuleDict

    def __init__(
        self,
        encoder: Encoder,
        y_classifier: Classifier,
        s_classifier: Classifier,
        method: Method,
        pseudo_labeler: PseudoLabeler,
        train_encoder: bool,
    ):
        classifier = nn.ModuleDict({"y": y_classifier, "s": s_classifier})
        super().__init__(
            encoder=encoder,
            classifier=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=train_encoder,
        )

    def supervised_loss(
        self, x: Tensor, s: Tensor, y: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> tuple[Tensor, LoggingDict]:
        total_loss = x.new_zeros(())
        logging_dict = defaultdict(float)
        # predict s and y separately
        for label, classifier in [(s, self.classifier["s"]), (y, self.classifier["y"])]:
            loss, logging = self.method.supervised_loss(
                encoder=self.encoder,
                classifier=cast(Classifier, classifier),
                x=x,
                class_id=label,
                ce_weight=ce_weight,
                bce_weight=bce_weight,
            )
            total_loss += loss
            for k, v in logging.items():
                logging_dict[k] += v  # logging dict only has losses, so taking sum is fine
        return total_loss, logging_dict

    def unsupervised_loss(self, x: Tensor) -> tuple[Tensor, LoggingDict]:
        joint, _, _, z = self._get_joint_output(x)
        return self.method.unsupervised_loss(
            pseudo_labeler=self.pseudo_labeler, z=z, preds=joint
        )

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.classifier["s"].step(grads)
        self.classifier["y"].step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

    def forward(self, x: Tensor) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        joint, s_logits, y_logits, _ = self._get_joint_output(x)
        return joint, (s_logits, y_logits)

    def _get_joint_output(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.encoder(x)
        y_logits = self.classifier["y"](z)
        s_logits = self.classifier["s"](z)
        y_probs = F.softmax(y_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)
        # take the outer product of s_probs and y_probs
        joint = s_probs[..., None] * y_probs[..., None, :]
        return joint.view(joint.shape[:-2] + (-1,)), s_logits, y_logits, z
