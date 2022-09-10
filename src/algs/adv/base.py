from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import Self, TypeAlias

from conduit.data.structures import NamedSample, TernarySample
from conduit.metrics import accuracy
from conduit.models.utils import prefix_keys
from loguru import logger
from ranzen import implements
from ranzen.torch import DcModule, cross_entropy_loss
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
import wandb

from src.algs.base import Algorithm
from src.arch.predictors.fcn import Fcn
from src.data import DataModule
from src.logging import log_images
from src.models.autoencoder import SplitLatentAe
from src.models.classifier import Classifier
from src.utils import to_item

from .evaluator import Evaluator

__all__ = ["AdvSemiSupervisedAlg", "Components"]


D = TypeVar("D")

IterTr: TypeAlias = Iterator[TernarySample[Tensor]]
IterDep: TypeAlias = Iterator[NamedSample[Tensor]]


@dataclass(eq=False)
class Components(DcModule, Generic[D]):
    ae: SplitLatentAe
    disc: D
    pred_y: Optional[Classifier]
    pred_s: Optional[Classifier]

    @torch.no_grad()
    def train_ae(self) -> None:
        self.ae.train()
        if self.pred_y is not None:
            self.pred_y.train()
        if self.pred_s is not None:
            self.pred_s.train()
        if isinstance(self.disc, nn.Module):
            self.disc.eval()

    @torch.no_grad()
    def train_disc(self) -> None:
        self.ae.eval()
        if self.pred_y is not None:
            self.pred_y.eval()
        if self.pred_s is not None:
            self.pred_s.eval()
        if isinstance(self.disc, nn.Module):
            self.disc.train()


@dataclass(eq=False)
class AdvSemiSupervisedAlg(Algorithm):
    """Base class for adversarial semi-supervsied methods."""

    _PBAR_COL: ClassVar[str] = "#ffe252"

    steps: int = 50_000
    # Number of gradient-accumulation steps
    ga_steps: int = 1
    weight_decay: float = 0
    warmup_steps: int = 0

    lr: float = 4.0e-4
    enc_loss_w: float = 1
    disc_loss_w: float = 1
    prior_loss_w: float = 0.0
    num_disc_updates: int = 3
    # Whether to use the deployment set when computing the encoder's adversarial loss
    twoway_disc_loss: bool = True

    pred_y_hidden_dim: Optional[int] = None
    pred_y_num_hidden: int = 0
    pred_y_loss_w: float = 1
    pred_s_loss_w: float = 0
    s_pred_with_bias: bool = False
    s_as_zs: bool = False

    predictor_y: Optional[Classifier] = field(init=False)
    predictor_s: Optional[Classifier] = field(init=False)

    # Misc
    validate: bool = True
    val_freq: Union[int, float] = 0.1  # how often to do validation
    log_freq: int = 150

    def __post_init__(self) -> None:
        if isinstance(self.val_freq, float) and (not (0 <= self.val_freq <= 1)):
            raise AttributeError("If 'val_freq' is a float, it must be in the range [0, 1].")
        super().__post_init__()

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
        pred_y = None
        if self.pred_y_loss_w > 0:
            model, _ = Fcn(hidden_dim=self.pred_y_hidden_dim,)(
                input_dim=ae.encoding_size.zy, target_dim=y_dim
            )[0]
            pred_y = Classifier(model=model, lr=self.lr).to(self.device)
        pred_s = None
        if self.pred_s_loss_w > 0:
            model, _ = Fcn(
                hidden_dim=None,  # no hidden layers
                final_bias=self.s_pred_with_bias,
            )(input_dim=ae.encoding_size.zs, target_dim=s_dim)
            pred_s = Classifier(model=model, lr=self.lr).to(self.device)

        return pred_y, pred_s

    @abstractmethod
    def discriminator_step(
        self, comp: Components, *, iterator_tr: IterTr, iterator_dep: IterDep
    ) -> None:
        ...

    def encoder_step(
        self,
        comp: Components,
        *,
        batch_tr: TernarySample,
        x_dep: Tensor,
        warmup: bool,
    ) -> DefaultDict[str, float]:
        logging_dict: DefaultDict[str, float] = defaultdict(float)
        for _ in range(self.ga_steps):
            loss, logging_dict_s = self._encoder_loss(
                comp=comp, x_dep=x_dep, batch_tr=batch_tr, warmup=warmup
            )
            self.backward(loss=loss / self.ga_steps)
            # Average the logging dict over the gradient-accumulation steps
            for k, v in logging_dict_s.items():
                logging_dict[k] = logging_dict[k] + (v / self.ga_steps)
        self._update_encoder(comp)
        return logging_dict

    def training_step(
        self,
        comp: Components,
        *,
        dm: DataModule,
        iterator_tr: IterTr,
        iterator_dep: IterDep,
        itr: int,
    ) -> Dict[str, float]:
        warmup = itr < self.warmup_steps
        if (not warmup) and (self.disc_loss_w > 0):
            comp.train_disc()
            self.discriminator_step(
                comp=comp,
                iterator_tr=iterator_tr,
                iterator_dep=iterator_dep,
            )

        batch_tr = self._sample_tr(iterator_tr=iterator_tr)
        x_dep = self._sample_dep(iterator_dep=iterator_dep)
        comp.train_ae()
        logging_dict = self.encoder_step(
            comp=comp,
            batch_tr=batch_tr,
            x_dep=x_dep,
            warmup=warmup,
        )
        logging_dict = prefix_keys(logging_dict, prefix="train", sep="/")  # type: ignore
        wandb.log(logging_dict, step=itr)

        # Log images
        if ((itr % self.log_freq) == 0) and (batch_tr.x.ndim == 4):
            with torch.no_grad():
                self.log_recons(x=batch_tr.x, dm=dm, ae=comp.ae, itr=itr, prefix="train")
                self.log_recons(x=x_dep, dm=dm, ae=comp.ae, itr=itr, prefix="deployment")
        return logging_dict

    @torch.no_grad()
    def log_recons(
        self,
        x: Tensor,
        *,
        dm: DataModule,
        ae: SplitLatentAe,
        itr: int,
        prefix: Optional[str] = None,
    ) -> None:
        """Log the reconstructed and original images."""
        num_blocks = min(4, len(x))
        rows_per_block = min(8, len(x) // num_blocks)
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = ae.encode(sample)
        recon = ae.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.s_as_zs:
            recons.append(recon.rand_s)
            caption += " | rand_s"

        to_log = [sample]
        for recon_ in recons:
            to_log.append(recon_)

        ncols = len(to_log)
        interleaved = torch.stack(to_log, dim=1).view(ncols * num_samples, *sample.shape[1:])

        log_images(
            images=interleaved,
            dm=dm,
            name="reconstructions",
            step=itr,
            nsamples=[ncols * rows_per_block] * num_blocks,
            ncols=ncols,
            prefix=prefix,
            caption=caption,
        )

    def backward(self, loss: Tensor) -> None:
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)  # type: ignore
        loss.backward()

    def _predictor_loss(
        self, comp: Components, *, zy: Tensor, zs: Tensor, y: Tensor, s: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss = torch.zeros((), device=self.device)
        logging_dict = {}
        if comp.pred_y is not None:
            # predictor is on encodings; predict y from the part that is invariant to s
            logits_pred_y = comp.pred_y(zy)
            pred_y_loss = cross_entropy_loss(input=logits_pred_y, target=y)
            pred_y_acc = accuracy(y_pred=logits_pred_y, y_true=y)

            pred_y_loss *= self.pred_y_loss_w
            logging_dict["Loss Predictor y"] = to_item(pred_y_loss)
            logging_dict["Accuracy Predictor y"] = to_item(pred_y_acc)
            loss += pred_y_loss
        if comp.pred_s is not None:
            logits_pred_s = comp.pred_s(zs)
            pred_s_loss = cross_entropy_loss(input=logits_pred_s, target=s)
            pred_s_acc = accuracy(y_pred=logits_pred_s, y_true=s)
            pred_s_loss *= self.pred_s_loss_w
            logging_dict["Loss Predictor s"] = to_item(pred_s_loss)
            logging_dict["Accuracy Predictor s"] = to_item(pred_s_acc)
            loss += pred_s_loss
        return loss, logging_dict

    @abstractmethod
    def _encoder_loss(
        self, comp: Components, *, x_dep: Tensor, batch_tr: TernarySample, warmup: bool
    ) -> Tuple[Tensor, Dict[str, float]]:
        ...

    def _update_encoder(self, comp: Components) -> None:
        # Clip the norm of the gradients if max_grad_norm is not None
        self._clip_gradients(comp.parameters())
        # Update the encoder's parameters
        comp.ae.step(grad_scaler=self.grad_scaler)
        comp.ae.zero_grad()
        if comp.pred_y is not None:
            comp.pred_y.step(grad_scaler=self.grad_scaler)
            comp.pred_y.zero_grad()
        if comp.pred_s is not None:
            comp.pred_s.step(grad_scaler=self.grad_scaler)
            comp.pred_s.zero_grad()
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _get_data_iterators(self, dm: DataModule) -> Tuple[IterTr, IterDep]:
        dl_tr = dm.train_dataloader()
        dl_dep = dm.deployment_dataloader()
        return iter(dl_tr), iter(dl_dep)

    def _evaluate(
        self, dm: DataModule, *, ae: SplitLatentAe, evaluator: Evaluator, step: Optional[int] = None
    ) -> None:
        if evaluator is not None:
            evaluator(dm=dm, encoder=ae, step=step, device=self.device)

    def fit(self, dm: DataModule, *, ae: SplitLatentAe, disc: Any, evaluator: Evaluator) -> Self:
        iterator_tr, iterator_dep = self._get_data_iterators(dm=dm)
        pred_y, pred_s = self._build_predictors(ae=ae, y_dim=dm.card_y, s_dim=dm.card_s)
        comp = Components(ae=ae, disc=disc, pred_y=pred_y, pred_s=pred_s)
        comp.to(self.device)

        val_freq = max(
            (
                self.val_freq
                if isinstance(self.val_freq, int)
                else round(self.val_freq * self.steps)
            ),
            1,
        )
        with tqdm(
            total=self.steps,
            desc="Training",
            colour=self._PBAR_COL,
        ) as pbar:
            for step in range(1, self.steps + 1):
                logging_dict = self.training_step(
                    comp=comp,
                    dm=dm,
                    iterator_tr=iterator_tr,
                    iterator_dep=iterator_dep,
                    itr=step,
                )
                pbar.set_postfix(logging_dict)
                pbar.update()

                if self.validate and (step % val_freq == 0):
                    self._evaluate(dm=dm, ae=ae, evaluator=evaluator, step=step)

        logger.info("Finished training")
        return self

    @implements(Algorithm)
    def run(
        self, dm: DataModule, *, ae: SplitLatentAe, disc: Any, evaluator: Evaluator, **kwargs: Any
    ) -> Any:
        try:
            self.fit(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
        except KeyboardInterrupt:
            ...
        logger.info("Performing final evaluation")
        self._evaluate(dm=dm, ae=ae, evaluator=evaluator, step=None)
