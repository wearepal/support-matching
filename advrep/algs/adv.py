from __future__ import annotations
from abc import abstractmethod
from collections.abc import Iterator
import logging
from pathlib import Path
import time
from typing import ClassVar
from typing_extensions import Literal, Self

from conduit.data.structures import NamedSample, TernarySample
from conduit.models.utils import prefix_keys
from hydra.utils import instantiate, to_absolute_path
from pandas.core.window.common import defaultdict
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
import wandb

from advrep.models import AutoEncoder, Classifier
from advrep.models.discriminator import SetDiscriminator
from advrep.optimisation import log_metrics, restore_model, save_model
from shared.configs import Config, DiscriminatorMethod
from shared.data import DataModule
from shared.models.configs import FcNet

from .base import Algorithm

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())

__all__ = ["AdvSemiSupervisedAlg"]


class AdvSemiSupervisedAlg(Algorithm):
    """Base class for adversarial semi-supervsied methods."""

    _PBAR_COL: ClassVar[str] = "#ffe252"

    encoder: AutoEncoder
    discriminator: SetDiscriminator
    predictor_y: Classifier | None
    predictor_s: Classifier | None

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg=cfg)
        self.enc_cfg = cfg.enc
        self.alg_cfg = cfg.alg
        self.adv_lr = self.alg_cfg.adv_lr
        self.grad_scaler = GradScaler() if self.use_amp else None

    def _sample_dep(self, iterator_dep: Iterator[NamedSample[Tensor]]) -> Tensor:
        return next(iterator_dep).x.to(self.device, non_blocking=True)

    def _sample_tr(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
    ) -> TernarySample[Tensor]:
        return next(iterator_tr).to(self.device, non_blocking=True)

    def _build_encoder(
        self,
        dm: DataModule,
    ) -> AutoEncoder:
        input_shape = dm.dim_x
        ae: AutoEncoder = instantiate(
            self.enc_cfg, input_shape=input_shape, feature_group_slices=dm.feature_group_slices
        )
        LOGGER.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        return ae

    @abstractmethod
    def _build_discriminator(self, encoder: AutoEncoder, *, dm: DataModule) -> SetDiscriminator:
        ...

    def _build_predictors(
        self, encoder: AutoEncoder, y_dim: int, *, s_dim: int
    ) -> tuple[Classifier | None, Classifier | None]:
        predictor_y = None
        if self.alg_cfg.pred_y_loss_w > 0:
            model = FcNet(
                hidden_dims=self.alg_cfg.pred_y_hidden_dims,
            )(input_dim=encoder.encoding_size.zy, target_dim=y_dim)
            predictor_y = Classifier(model=model, lr=self.adv_lr)
        predictor_s = None
        if self.alg_cfg.pred_s_loss_w > 0:
            model = FcNet(
                hidden_dims=None,  # no hidden layers
                final_layer_bias=self.alg_cfg.s_pred_with_bias,
            )(input_dim=encoder.encoding_size.zs, target_dim=s_dim)
            predictor_s = Classifier(model=model, lr=self.adv_lr)

        return predictor_y, predictor_s

    def _build(self, dm: DataModule) -> None:
        self.encoder = self._build_encoder(dm=dm)
        self.discriminator = self._build_discriminator(encoder=self.encoder, dm=dm)
        self.predictor_y, self.predictor_s = self._build_predictors(
            encoder=self.encoder, y_dim=dm.card_y, s_dim=dm.card_s
        )
        self.to(self.device)

    def training_step(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
        itr: int,
        dm: DataModule,
    ) -> dict[str, float]:
        warmup = itr < self.alg_cfg.warmup_steps
        ga_weight = 1 / self.alg_cfg.num_adv_updates
        if (not warmup) and (self.alg_cfg.adv_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.alg_cfg.num_adv_updates):
                for _ in range(self.alg_cfg.ga_steps):
                    loss, _ = self._discriminator_loss(
                        iterator_tr=iterator_tr, iterator_dep=iterator_dep
                    )
                    self.backward(loss / ga_weight)
                self._update_discriminator()

        batch_tr = self._sample_tr(iterator_tr=iterator_tr)
        x_dep = self._sample_dep(iterator_dep=iterator_dep)
        logging_dict: dict[str, float] = defaultdict(float)
        for _ in range(self.alg_cfg.ga_steps):
            loss, logging_dict_s = self._encoder_loss(x_dep=x_dep, batch_tr=batch_tr, warmup=warmup)
            self.backward(loss / ga_weight)
            # Average the logging dict over the gradient-accumulation steps
            for k, v in logging_dict_s.items():
                logging_dict[k] = logging_dict[k] + (v / self.alg_cfg.ga_steps)
        self._update_encoder()

        logging_dict = prefix_keys(logging_dict, prefix="train", sep="/")
        wandb.log(logging_dict, step=itr)

        # Log images
        if ((itr % self.alg_cfg.log_freq) == 0) and (batch_tr.x.ndim == 4):
            with torch.no_grad():
                self.log_recons(x=batch_tr.x, dm=dm, itr=itr, prefix="train")
                self.log_recons(x=x_dep, dm=dm, itr=itr, prefix="deployment")
        return logging_dict

    @abstractmethod
    @torch.no_grad()
    def log_recons(self, x: Tensor, *, dm: DataModule, itr: int, prefix: str | None = None) -> None:
        ...

    def backward(self, loss: Tensor) -> None:
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)  # type: ignore
        loss.backward()

    @abstractmethod
    def _discriminator_loss(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    def _clip_gradients(self, parameters: Iterator[Parameter]) -> None:
        if (value := self.alg_cfg.max_grad_norm) is not None:
            nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=value, norm_type=2.0)

    @abstractmethod
    def _encoder_loss(
        self, x_dep: Tensor, *, batch_tr: TernarySample, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    def _update_encoder(self) -> None:
        # Clip the norm of the gradients if max_grad_norm is not None
        self._clip_gradients(self.discriminator.parameters())
        # Update the encoder's parameters
        self.encoder.step(grad_scaler=self.grad_scaler)
        if self.predictor_y is not None:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.predictor_s is not None:
            self.predictor_s.step(grad_scaler=self.grad_scaler)

        self.encoder.zero_grad()
        if self.predictor_y is not None:
            self.predictor_y.zero_grad()
        if self.predictor_s is not None:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _update_discriminator(self) -> None:
        self._clip_gradients(self.discriminator.parameters())
        self.discriminator.step(grad_scaler=self.grad_scaler)
        self.discriminator.zero_grad()
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _train(self, mode: Literal["encoder", "discriminator"]) -> None:
        if mode == "encoder":
            self.encoder.train()
            if self.predictor_y is not None:
                self.predictor_y.train()
            if self.predictor_s is not None:
                self.predictor_s.train()
            self.discriminator.eval()
        else:
            self.encoder.eval()
            if self.predictor_y is not None:
                self.predictor_y.eval()
            if self.predictor_s is not None:
                self.predictor_s.eval()
            self.discriminator.train()

    def _get_data_iterators(
        self, dm: DataModule, *, group_ids: Tensor | None = None
    ) -> tuple[Iterator[TernarySample[Tensor]], Iterator[NamedSample[Tensor]]]:
        dl_tr = dm.train_dataloader()
        dl_dep = dm.deployment_dataloader(group_ids=group_ids)

        return iter(dl_tr), iter(dl_dep)

    def fit(self, dm: DataModule, *, group_ids: Tensor | None = None) -> Self:
        # Construct the data iterators
        iterator_tr, iterator_dep = self._get_data_iterators(dm=dm, group_ids=group_ids)
        # ==== construct networks ====
        self._build(dm)

        save_dir = Path(to_absolute_path(self.log_cfg.save_dir)) / str(time.time())
        save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Save directory: {save_dir.resolve()}")

        start_itr = 1  # start at 1 so that the val_freq works correctly
        # Resume from checkpoint
        if self.misc_cfg.resume is not None:
            LOGGER.info("Restoring encoder's weights from checkpoint")
            encoder, start_itr = restore_model(self.cfg, Path(self.misc_cfg.resume), self.encoder)

            if self.misc_cfg.evaluate:
                log_metrics(
                    cfg=self.cfg,
                    encoder=encoder,
                    dm=dm,
                    step=0,
                    save_summary=True,
                    device=self.device,
                )

        itr = start_itr
        with tqdm(
            total=self.alg_cfg.iters - start_itr,
            desc="Training",
            colour=self._PBAR_COL,
        ) as pbar:
            for itr in range(start_itr, self.alg_cfg.iters + 1):
                logging_dict = self.training_step(
                    iterator_tr=iterator_tr,
                    iterator_dep=iterator_dep,
                    itr=itr,
                    dm=dm,
                )
                pbar.set_postfix(logging_dict)
                pbar.update()

                if self.alg_cfg.validate and itr % self.alg_cfg.val_freq == 0:
                    log_metrics(self.cfg, encoder=self.encoder, dm=dm, step=itr, device=self.device)
                    save_model(self.cfg, save_dir, model=self.encoder, itr=itr)

        LOGGER.info("Training has finished.")

        log_metrics(
            self.cfg,
            encoder=self.encoder,
            dm=dm,
            save_summary=True,
            step=itr,
            cluster_metrics=None,
            device=self.device,
        )
        return self
