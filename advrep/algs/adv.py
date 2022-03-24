from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator
import logging
from pathlib import Path
import time
from typing_extensions import Literal, Self

from conduit.data.structures import NamedSample, TernarySample
from hydra.utils import to_absolute_path
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.nn.functional as F
import wandb

from advrep.models import AutoEncoder, Classifier, EncodingSize, build_classifier
from advrep.models.discriminator import Discriminator
from advrep.optimisation import (
    MixedLoss,
    build_ae,
    log_metrics,
    restore_model,
    save_model,
)
from shared.configs import Config, DiscriminatorMethod, ReconstructionLoss
from shared.data import DataModule
from shared.models.configs import FcNet, conv_autoencoder, fc_autoencoder
from shared.utils import AverageMeter, ClusterResults, readable_duration

from .base import Algorithm

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())

__all__ = ["AdvSemiSupervisedAlg"]


class AdvSemiSupervisedAlg(Algorithm):
    """Base class for adversarial semi-supervsied methods."""

    _encoding_size: EncodingSize
    encoder: AutoEncoder
    adversary: Discriminator
    recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
    predictor_y: Classifier | None
    predictor_s: Classifier | None

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg=cfg)
        self.enc_cfg = cfg.enc
        self.alg_cfg = cfg.alg
        self.optimizer_kwargs = {"lr": self.alg_cfg.adv_lr}
        self.grad_scaler = GradScaler() if self.train_cfg.use_amp else None

    def _sample_dep(self, iterator_dep: Iterator[NamedSample[Tensor]]) -> Tensor:
        return next(iterator_dep).to(self.device).x

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
        s_dim = dm.card_s
        if len(input_shape) > 2:
            encoder, decoder, latent_dim = conv_autoencoder(
                input_shape,
                initial_hidden_channels=self.enc_cfg.init_chans,
                encoding_dim=self.enc_cfg.out_dim,
                decoding_dim=input_shape[0],
                levels=self.enc_cfg.levels,
                decoder_out_act=None,
                variational=self.alg_cfg.vae,
            )
        else:
            encoder, decoder, latent_dim = fc_autoencoder(
                input_shape,
                hidden_channels=self.enc_cfg.init_chans,
                encoding_dim=self.enc_cfg.out_dim,
                levels=self.enc_cfg.levels,
                variational=self.alg_cfg.vae,
            )

        zs_dim = self.alg_cfg.zs_dim
        zy_dim = latent_dim - zs_dim
        self._encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)

        LOGGER.info(f"Encoding dim: {latent_dim}, {self._encoding_size}")

        encoder = build_ae(
            cfg=self.cfg,
            encoder=encoder,
            decoder=decoder,
            encoding_size=self._encoding_size,
            s_dim=s_dim,
            feature_group_slices=dm.feature_group_slices,
        )

        # load pretrained encoder if one is provided
        if self.enc_cfg.checkpoint_path:
            save_dict = torch.load(self.enc_cfg.checkpoint_path, map_location=self.device)
            encoder.load_state_dict(save_dict["encoder"])

        return encoder

    @abstractmethod
    def _build_adversary(self, dm: DataModule) -> Discriminator:
        ...

    def _build_predictors(
        self, y_dim: int, *, s_dim: int
    ) -> tuple[Classifier | None, Classifier | None]:
        predictor_y = None
        if self.alg_cfg.pred_y_loss_w > 0:
            predictor_y = build_classifier(
                input_shape=(self._encoding_size.zy,),  # this is always trained on encodings
                target_dim=y_dim,
                model_fn=FcNet(hidden_dims=self.alg_cfg.pred_y_hidden_dims),
                optimizer_kwargs=self.optimizer_kwargs,
            )
        predictor_s = None
        if self.alg_cfg.pred_s_loss_w > 0:
            predictor_s = build_classifier(
                input_shape=(self._encoding_size.zs,),  # this is always trained on encodings
                target_dim=s_dim,
                model_fn=FcNet(
                    hidden_dims=None,  # no hidden layers
                    final_layer_bias=self.alg_cfg.s_pred_with_bias,
                ),
                optimizer_kwargs=self.optimizer_kwargs,
            )
        return predictor_y, predictor_s

    def _build(self, dm: DataModule) -> None:
        self.encoder = self._build_encoder(dm=dm)
        self.recon_loss_fn = self._get_recon_loss_fn(feature_group_slices=dm.feature_group_slices)
        self.adversary = self._build_adversary(dm=dm)
        self.predictor_y, self.predictor_s = self._build_predictors(
            y_dim=dm.card_y, s_dim=dm.card_s
        )
        self.to(self.train_cfg.device)

    def _get_recon_loss_fn(
        self, feature_group_slices: dict[str, list[slice]] | None = None
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if self.enc_cfg.recon_loss is ReconstructionLoss.l1:
            recon_loss_fn = nn.L1Loss(reduction="sum")
        elif self.enc_cfg.recon_loss is ReconstructionLoss.l2:
            recon_loss_fn = nn.MSELoss(reduction="sum")
        elif self.enc_cfg.recon_loss is ReconstructionLoss.bce:
            recon_loss_fn = nn.BCELoss(reduction="sum")
        elif self.enc_cfg.recon_loss is ReconstructionLoss.huber:
            recon_loss_fn = lambda x, y: 0.1 * F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
        elif self.enc_cfg.recon_loss is ReconstructionLoss.mixed:
            assert (
                feature_group_slices is not None
            ), "can only do multi enc_loss with feature groups"
            recon_loss_fn = MixedLoss(feature_group_slices, reduction="sum")
        else:
            raise ValueError(f"{self.enc_cfg.recon_loss} is an invalid reconstruction loss.")
        return recon_loss_fn

    def _train_step(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
        itr: int,
        dm: DataModule,
    ) -> dict[str, float]:

        warmup = itr < self.alg_cfg.warmup_steps
        if (not warmup) and (self.alg_cfg.adv_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.alg_cfg.num_adv_updates):
                self._step_adversary(iterator_tr=iterator_tr, iterator_dep=iterator_dep)

        batch_tr = self._sample_tr(iterator_tr=iterator_tr)
        x_dep = self._sample_dep(iterator_dep=iterator_dep)
        _, logging_dict = self._step_encoder(x_dep=x_dep, batch_tr=batch_tr, warmup=warmup)

        wandb.log(logging_dict, step=itr)

        # Log images
        if ((itr % self.alg_cfg.log_freq) == 0) and (batch_tr.x.ndim == 4):
            self._log_recons(x=batch_tr.x, dm=dm, itr=itr, prefix="train")
            self._log_recons(x=x_dep, dm=dm, itr=itr, prefix="deployment")
        return logging_dict

    @abstractmethod
    @torch.no_grad()
    def _log_recons(
        self, x: Tensor, *, dm: DataModule, itr: int, prefix: str | None = None
    ) -> None:
        ...

    @abstractmethod
    def _step_adversary(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    @abstractmethod
    def _step_encoder(
        self, x_dep: Tensor, *, batch_tr: TernarySample, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    def _update_encoder(self, loss: Tensor) -> None:
        self.encoder.zero_grad()
        if self.predictor_y is not None:
            self.predictor_y.zero_grad()
        if self.predictor_s is not None:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)  # type: ignore
        loss.backward()

        # Update the encoder's parameters
        self.encoder.step(grad_scaler=self.grad_scaler)
        if self.predictor_y is not None:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.predictor_s is not None:
            self.predictor_s.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _update_adversary(self, loss: Tensor) -> None:
        self.adversary.zero_grad()
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)  # type: ignore
        loss.backward()

        self.adversary.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    def _train(self, mode: Literal["encoder", "adversary"]) -> None:
        if mode == "encoder":
            self.encoder.train()
            if self.predictor_y is not None:
                self.predictor_y.train()
            if self.predictor_s is not None:
                self.predictor_s.train()
            self.adversary.eval()
        else:
            self.encoder.eval()
            if self.predictor_y is not None:
                self.predictor_y.eval()
            if self.predictor_s is not None:
                self.predictor_s.eval()
            self.adversary.train()

    def _get_data_iterators(
        self, dm: DataModule, *, cluster_results: ClusterResults | None = None
    ) -> tuple[Iterator[TernarySample[Tensor]], Iterator[NamedSample[Tensor]]]:
        dl_tr = dm.train_dataloader()
        dl_dep = dm.deployment_dataloader(cluster_results=cluster_results)

        return iter(dl_tr), iter(dl_dep)

    def fit(self, dm: DataModule, cluster_results: ClusterResults | None = None) -> Self:
        # Construct the data iterators
        iterator_tr, iterator_dep = self._get_data_iterators(dm=dm, cluster_results=cluster_results)
        # ==== construct networks ====
        self._build(dm)

        save_dir = Path(to_absolute_path(self.log_cfg.save_dir)) / str(time.time())
        save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Save directory: {save_dir.resolve()}")

        start_itr = 1  # start at 1 so that the val_freq works correctly
        # Resume from checkpoint
        if self.train_cfg.resume is not None:
            LOGGER.info("Restoring encoder's weights from checkpoint")
            encoder, start_itr = restore_model(self.cfg, Path(self.train_cfg.resume), self.encoder)

            if self.train_cfg.evaluate:
                log_metrics(
                    cfg=self.cfg,
                    encoder=encoder,
                    dm=dm,
                    step=0,
                    save_summary=True,
                    cluster_metrics=None,
                )

        itr = start_itr
        start_time = time.monotonic()
        loss_meters = defaultdict(AverageMeter)

        for itr in range(start_itr, self.alg_cfg.iters + 1):
            logging_dict = self._train_step(
                iterator_tr=iterator_tr,
                iterator_dep=iterator_dep,
                itr=itr,
                dm=dm,
            )
            for name, value in logging_dict.items():
                loss_meters[name].update(value)

            if itr % self.alg_cfg.log_freq == 0:
                log_string = " | ".join(
                    f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items()
                )
                elapsed = time.monotonic() - start_time
                LOGGER.info(
                    f"[TRN] Iteration {itr} | Elapsed: {readable_duration(elapsed)} | "
                    f"Iterations/s: {self.alg_cfg.log_freq / elapsed} | {log_string}"
                )

                loss_meters.clear()
                start_time = time.monotonic()

            if self.alg_cfg.validate and itr % self.alg_cfg.val_freq == 0:
                log_metrics(self.cfg, encoder=self.encoder, dm=dm, step=itr)
                save_model(self.cfg, save_dir, model=self.encoder, itr=itr)

        LOGGER.info("Training has finished.")

        log_metrics(
            self.cfg,
            encoder=self.encoder,
            dm=dm,
            save_summary=True,
            step=itr,
            cluster_metrics=None,
        )
        return self
