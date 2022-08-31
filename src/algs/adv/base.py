from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, DefaultDict, Dict, Iterator, Optional, Tuple, Union
from typing_extensions import Literal, Self

from conduit.data.structures import NamedSample, TernarySample
from conduit.models.utils import prefix_keys
from loguru import logger
from omegaconf import MISSING
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
import wandb

from src.algs.base import Algorithm
from src.arch.predictors.fcn import Fcn
from src.data import DataModule
from src.models.autoencoder import SplitLatentAe
from src.models.classifier import Classifier
from src.models.discriminator import MmdDiscriminator, NeuralDiscriminator

from .evaluator import Evaluator

__all__ = ["AdvSemiSupervisedAlg"]


@dataclass
class AdvSemiSupervisedAlg(Algorithm):
    """Base class for adversarial semi-supervsied methods."""

    _PBAR_COL: ClassVar[str] = "#ffe252"

    ae: SplitLatentAe = MISSING
    disc: Union[NeuralDiscriminator, MmdDiscriminator] = MISSING
    evaluator: Optional[Evaluator] = MISSING

    steps: int = 50_000
    # Gradient-accumulation steps
    ga_steps: int = 1
    weight_decay: float = 0
    warmup_steps: int = 0
    max_grad_norm: Optional[float] = None

    lr: float = 4.0e-4
    enc_loss_w: float = 1
    disc_loss_w: float = 1
    num_disc_updates: int = 3
    # Whether to use the deployment set when computing the encoder's adversarial loss
    twoway_disc_loss: bool = True

    pred_y_hidden_dim: Optional[int] = None
    pred_y_num_hidden: int = 0
    pred_y_loss_w: float = 1
    pred_s_loss_w: float = 0
    s_pred_with_bias: bool = False

    predictor_y: Optional[Classifier] = field(init=False)
    predictor_s: Optional[Classifier] = field(init=False)

    # Misc
    validate: bool = True
    val_freq: int = 1_000  # how often to do validation
    log_freq: int = 150

    def _sample_dep(self, iterator_dep: Iterator[NamedSample[Tensor]]) -> Tensor:
        return next(iterator_dep).x.to(self.device, non_blocking=True)

    def _sample_tr(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
    ) -> TernarySample[Tensor]:
        return next(iterator_tr).to(self.device, non_blocking=True)

    def _build_predictors(
        self, ae: SplitLatentAe, *, y_dim: int, s_dim: int
    ) -> Tuple[Optional[Classifier], Optional[Classifier]]:
        predictor_y = None
        if self.pred_y_loss_w > 0:
            model, _ = Fcn(hidden_dim=self.pred_y_hidden_dim,)(
                input_dim=ae.encoding_size.zy, target_dim=y_dim
            )[0]
            predictor_y = Classifier(model=model, lr=self.lr)
        predictor_s = None
        if self.pred_s_loss_w > 0:
            model, _ = Fcn(
                hidden_dim=None,  # no hidden layers
                final_bias=self.s_pred_with_bias,
            )(input_dim=ae.encoding_size.zs, target_dim=s_dim)
            predictor_s = Classifier(model=model, lr=self.lr)

        return predictor_y, predictor_s

    def training_step(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
        itr: int,
        dm: DataModule,
    ) -> Dict[str, float]:
        warmup = itr < self.warmup_steps
        ga_weight = 1 / self.num_disc_updates
        if (not warmup) and (isinstance(self.disc, NeuralDiscriminator)):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.num_disc_updates):
                for _ in range(self.ga_steps):
                    loss, _ = self._discriminator_loss(
                        iterator_tr=iterator_tr, iterator_dep=iterator_dep
                    )
                    self.backward(loss / ga_weight)
                self._update_discriminator()

        batch_tr = self._sample_tr(iterator_tr=iterator_tr)
        x_dep = self._sample_dep(iterator_dep=iterator_dep)
        logging_dict: DefaultDict[str, float] = defaultdict(float)
        for _ in range(self.ga_steps):
            loss, logging_dict_s = self._encoder_loss(x_dep=x_dep, batch_tr=batch_tr, warmup=warmup)
            self.backward(loss / ga_weight)
            # Average the logging dict over the gradient-accumulation steps
            for k, v in logging_dict_s.items():
                logging_dict[k] = logging_dict[k] + (v / self.ga_steps)
        self._update_encoder()

        logging_dict = prefix_keys(logging_dict, prefix="train", sep="/")  # type: ignore
        wandb.log(logging_dict, step=itr)

        # Log images
        if ((itr % self.log_freq) == 0) and (batch_tr.x.ndim == 4):
            with torch.no_grad():
                self.log_recons(x=batch_tr.x, dm=dm, itr=itr, prefix="train")
                self.log_recons(x=x_dep, dm=dm, itr=itr, prefix="deployment")
        return logging_dict

    @abstractmethod
    @torch.no_grad()
    def log_recons(
        self, x: Tensor, *, dm: DataModule, itr: int, prefix: Optional[str] = None
    ) -> None:
        ...

    @abstractmethod
    def _encoder_loss(
        self, x_dep: Tensor, *, batch_tr: TernarySample, warmup: bool
    ) -> Tuple[Tensor, Dict[str, float]]:
        ...

    @abstractmethod
    def _discriminator_loss(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
    ) -> Tuple[Tensor, Dict[str, float]]:
        ...

    def backward(self, loss: Tensor) -> None:
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)  # type: ignore
        loss.backward()

    def _clip_gradients(self, parameters: Iterator[Parameter]) -> None:
        if (value := self.max_grad_norm) is not None:
            nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=value, norm_type=2.0)

    def _update_encoder(self) -> None:
        # Clip the norm of the gradients if max_grad_norm is not None
        self._clip_gradients(self.ae.parameters())
        # Update the encoder's parameters
        self.ae.step(grad_scaler=self.grad_scaler)
        if self.predictor_y is not None:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.predictor_s is not None:
            self.predictor_s.step(grad_scaler=self.grad_scaler)

        self.ae.zero_grad()
        if self.predictor_y is not None:
            self.predictor_y.zero_grad()
        if self.predictor_s is not None:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _update_discriminator(self) -> None:
        if isinstance(self.disc, NeuralDiscriminator):
            self._clip_gradients(self.disc.parameters())
            self.disc.step(grad_scaler=self.grad_scaler)
            self.disc.zero_grad()
            if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
                self.grad_scaler.update()

    def _train(self, mode: Literal["encoder", "discriminator"]) -> None:
        if mode == "encoder":
            self.ae.train()
            if self.predictor_y is not None:
                self.predictor_y.train()
            if self.predictor_s is not None:
                self.predictor_s.train()
            self.disc.eval()
        else:
            self.ae.eval()
            if self.predictor_y is not None:
                self.predictor_y.eval()
            if self.predictor_s is not None:
                self.predictor_s.eval()
            self.disc.train()

    def _get_data_iterators(
        self, dm: DataModule
    ) -> Tuple[Iterator[TernarySample[Tensor]], Iterator[NamedSample[Tensor]]]:
        dl_tr = dm.train_dataloader()
        dl_dep = dm.deployment_dataloader()
        return iter(dl_tr), iter(dl_dep)

    def _evaluate(self, dm: DataModule, step: int) -> None:
        if self.evaluator is not None:
            self.evaluator(dm=dm, encoder=self.ae, step=step, device=self.device)

    def fit(self, dm: DataModule) -> Self:
        # Construct the data iterators
        iterator_tr, iterator_dep = self._get_data_iterators(dm=dm)
        # ==== construct networks ====
        self.predictor_y, self.predictor_s = self._build_predictors(
            ae=self.ae, y_dim=dm.card_y, s_dim=dm.card_s
        )
        self.to(self.device)

        start_itr = 1  # start at 1 so that the val_freq works correctly
        step = start_itr
        with tqdm(
            total=self.steps - start_itr,
            desc="Training",
            colour=self._PBAR_COL,
        ) as pbar:
            for step in range(start_itr, self.steps + 1):
                logging_dict = self.training_step(
                    iterator_tr=iterator_tr,
                    iterator_dep=iterator_dep,
                    itr=step,
                    dm=dm,
                )
                pbar.set_postfix(logging_dict)
                pbar.update()

                if self.validate and (step % self.val_freq == 0):
                    self._evaluate(dm=dm, step=step)

        self._evaluate(dm=dm, step=step)
        logger.info("Training has finished.")
        return self
