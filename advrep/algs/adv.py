from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator
import logging
from pathlib import Path
import time
from typing import Any, cast
from typing_extensions import Literal

from hydra.utils import to_absolute_path
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from advrep.models import (
    AutoEncoder,
    Classifier,
    Discriminator,
    EncodingSize,
    build_classifier,
)
from advrep.optimisation import (
    MixedLoss,
    PixelCrossEntropy,
    baseline_metrics,
    build_ae,
    build_weighted_sampler_from_dataset,
    get_stratified_sampler,
    log_metrics,
    restore_model,
    save_model,
)
from shared.configs import (
    Config,
    DiscriminatorMethod,
    ImageDatasetConfig,
    ReconstructionLoss,
)
from shared.data import Batch, DataModule, RandomSampler
from shared.models.configs import FcNet, conv_autoencoder, fc_autoencoder
from shared.utils import (
    AverageMeter,
    ClusterResults,
    inf_generator,
    load_results,
    readable_duration,
)

from .base import AlgBase

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())

__all__ = ["AdvSemiSupervisedAlg"]


class AdvSemiSupervisedAlg(AlgBase):
    """Base class for adversarial semi-supervsied methods."""

    _encoding_size: EncodingSize
    encoder: AutoEncoder
    adversary: Classifier | Discriminator
    recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
    predictor_y: Classifier | None
    predictor_s: Classifier | None

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg=cfg)
        self.enc_cfg = cfg.enc
        self.adapt_cfg = cfg.adapt
        self.optimizer_kwargs = {"lr": self.adapt_cfg.adv_lr}
        self.eff_batch_size = self.adapt_cfg.batch_size
        self.grad_scaler = GradScaler() if self.misc_cfg.use_amp else None

    def _sample_context(self, context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        return cast(Tensor, self._to_device(next(context_data_itr)[0]))

    def _sample_train(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> Batch:
        x_tr, s_tr, y_tr = self._to_device(*next(train_data_itr))
        return Batch(x_tr, s_tr, y_tr)

    def _build_encoder(
        self,
        input_shape: tuple[int, ...],
        s_dim: int,
        feature_group_slices: dict[str, list[slice]] | None = None,
    ) -> AutoEncoder:
        if len(input_shape) > 2:
            decoding_dim = (
                input_shape[0] * 256
                if self.enc_cfg.recon_loss is ReconstructionLoss.ce
                else input_shape[0]
            )
            decoder_out_act = None
            encoder, decoder, latent_dim = conv_autoencoder(
                input_shape,
                self.enc_cfg.init_chans,
                encoding_dim=self.enc_cfg.out_dim,
                decoding_dim=decoding_dim,
                levels=self.enc_cfg.levels,
                decoder_out_act=decoder_out_act,
                variational=self.adapt_cfg.vae,
            )
        else:
            encoder, decoder, latent_dim = fc_autoencoder(
                input_shape,
                self.enc_cfg.init_chans,
                encoding_dim=self.enc_cfg.out_dim,
                levels=self.enc_cfg.levels,
                variational=self.adapt_cfg.vae,
            )

        zs_dim = self.adapt_cfg.zs_dim
        zy_dim = latent_dim - zs_dim
        self._encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)

        LOGGER.info(f"Encoding dim: {latent_dim}, {self._encoding_size}")

        encoder = build_ae(
            cfg=self.cfg,
            encoder=encoder,
            decoder=decoder,
            encoding_size=self._encoding_size,
            s_dim=s_dim,
            feature_group_slices=feature_group_slices,
        )

        # load pretrained encoder if one is provided
        if self.enc_cfg.checkpoint_path:
            save_dict = torch.load(
                self.enc_cfg.checkpoint_path, map_location=lambda storage, loc: storage
            )
            encoder.load_state_dict(save_dict["encoder"])

        return encoder

    @abstractmethod
    def _build_adversary(
        self, input_shape: tuple[int, ...], s_dim: int
    ) -> Classifier | Discriminator:
        ...

    def _build_predictors(
        self, y_dim: int, s_dim: int
    ) -> tuple[Classifier | None, Classifier | None]:
        predictor_y = None
        if self.adapt_cfg.pred_y_loss_w > 0:
            predictor_y = build_classifier(
                input_shape=(self._encoding_size.zy,),  # this is always trained on encodings
                target_dim=y_dim,
                model_fn=FcNet(hidden_dims=self.adapt_cfg.pred_y_hidden_dims),
                optimizer_kwargs=self.optimizer_kwargs,
            )
        predictor_s = None
        if self.adapt_cfg.pred_s_loss_w > 0:
            predictor_s = build_classifier(
                input_shape=(self._encoding_size.zs,),  # this is always trained on encodings
                target_dim=s_dim,
                model_fn=FcNet(
                    hidden_dims=None,  # no hidden layers
                    final_layer_bias=self.adapt_cfg.s_pred_with_bias,
                ),
                optimizer_kwargs=self.optimizer_kwargs,
            )
        return predictor_y, predictor_s

    def _build(
        self,
        input_shape: tuple[int, ...],
        y_dim: int,
        s_dim: int,
        feature_group_slices: dict[str, list[slice]] | None = None,
    ) -> None:
        self.encoder = self._build_encoder(
            input_shape=input_shape, s_dim=s_dim, feature_group_slices=feature_group_slices
        )
        self.recon_loss_fn = self._get_recon_loss_fn(feature_group_slices=feature_group_slices)
        self.adversary = self._build_adversary(input_shape=input_shape, s_dim=s_dim)
        self.predictor_y, self.predictor_s = self._build_predictors(y_dim=y_dim, s_dim=s_dim)
        self.to(self.misc_cfg.device)

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
        elif self.enc_cfg.recon_loss is ReconstructionLoss.ce:
            recon_loss_fn = PixelCrossEntropy(reduction="sum")
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
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:

        warmup = itr < self.adapt_cfg.warmup_steps
        if (not warmup) and (self.adapt_cfg.adv_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.adapt_cfg.num_adv_updates):
                self._step_adversary(
                    train_data_itr=train_data_itr, context_data_itr=context_data_itr
                )

        batch_tr = self._sample_train(train_data_itr=train_data_itr)
        x_ctx = self._sample_context(context_data_itr=context_data_itr)
        _, logging_dict = self._step_encoder(x_ctx=x_ctx, batch_tr=batch_tr, warmup=warmup)

        wandb.log(logging_dict, step=itr)

        # Log images
        if itr % self.adapt_cfg.log_freq == 0 and isinstance(self.data_cfg, ImageDatasetConfig):
            with torch.no_grad():
                self._log_recons(x=batch_tr.x, itr=itr, prefix="train")
                self._log_recons(x=x_ctx, itr=itr, prefix="context")
        return logging_dict

    @abstractmethod
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        ...

    @abstractmethod
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    @abstractmethod
    def _step_encoder(
        self, x_ctx: Tensor, batch_tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    def _update_encoder(self, loss: Tensor) -> None:
        self.encoder.zero_grad()
        if self.predictor_y is not None:
            self.predictor_y.zero_grad()
        if self.predictor_s is not None:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)
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
            loss = self.grad_scaler.scale(loss)
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
        self, datasets: DataModule, cluster_results: ClusterResults | None = None
    ) -> tuple[Iterator[Batch], Iterator[Batch]]:
        s_count = max(datasets.dim_s, 2)
        context_dl_kwargs: dict[str, Any] = dict(shuffle=False, drop_last=True)
        if cluster_results is not None:
            context_sampler = get_stratified_sampler(
                group_ids=cluster_results.cluster_ids,
                oversample=self.adapt_cfg.oversample,
                batch_size=self.adapt_cfg.batch_size,
                min_size=None if self.adapt_cfg.oversample else self.eff_batch_size,
            )
            if self.adapt_cfg.use_pretrained_enc:
                self.enc_cfg.checkpoint_path = str(cluster_results.enc_path)
        elif self.adapt_cfg.balanced_context:
            context_sampler = build_weighted_sampler_from_dataset(
                dataset=datasets.context,  # type: ignore
                s_count=s_count,
                batch_size=self.eff_batch_size,
                oversample=self.adapt_cfg.oversample,
                balance_hierarchical=False,
            )
        else:
            context_sampler = RandomSampler(datasets.context)  # type: ignore
        context_dl_kwargs["sampler"] = context_sampler

        context_dataloader = DataLoader(
            datasets.context,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            batch_size=self.eff_batch_size,
            **context_dl_kwargs,
        )

        train_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.train,  # type: ignore
            s_count=s_count,
            batch_size=self.eff_batch_size,
            oversample=self.adapt_cfg.oversample,
            balance_hierarchical=True,
        )
        train_dataloader = DataLoader(
            dataset=datasets.train,
            num_workers=self.data_cfg.num_workers,
            drop_last=True,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=True,
            batch_size=self.eff_batch_size,
        )
        train_data_itr = inf_generator(train_dataloader)
        context_data_itr = inf_generator(context_dataloader)

        return train_data_itr, context_data_itr

    def _fit(self, datasets: DataModule) -> None:
        # Load cluster results
        cluster_results = None
        cluster_metrics: dict[str, float] | None = None
        if self.misc_cfg.cluster_label_file:
            cluster_results, cluster_metrics = load_results(self.cfg)

        # Construct the data iterators
        train_data_itr, context_data_itr = self._get_data_iterators(
            datasets=datasets, cluster_results=cluster_results
        )
        # ==== construct networks ====
        input_shape = next(context_data_itr)[0][0].shape
        feature_group_slices = getattr(datasets.context, "feature_group_slices", None)
        self._build(
            input_shape=input_shape,
            y_dim=datasets.dim_y,
            s_dim=datasets.dim_s,
            feature_group_slices=feature_group_slices,
        )

        save_dir = Path(to_absolute_path(self.misc_cfg.save_dir)) / str(time.time())
        save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Save directory: {save_dir.resolve()}")

        start_itr = 1  # start at 1 so that the val_freq works correctly
        # Resume from checkpoint
        if self.misc_cfg.resume is not None:
            LOGGER.info("Restoring encoder's weights from checkpoint")
            encoder, start_itr = restore_model(self.cfg, Path(self.misc_cfg.resume), self.encoder)
            self.encoder = cast(AutoEncoder, encoder)

            if self.misc_cfg.evaluate:
                log_metrics(
                    self.cfg,
                    encoder,
                    datasets,
                    step=0,
                    save_summary=True,
                    cluster_metrics=cluster_metrics,
                )

        itr = start_itr
        start_time = time.monotonic()
        loss_meters = defaultdict(AverageMeter)

        for itr in range(start_itr, self.adapt_cfg.iters + 1):

            logging_dict = self._train_step(
                context_data_itr=context_data_itr,
                train_data_itr=train_data_itr,
                itr=itr,
            )
            for name, value in logging_dict.items():
                loss_meters[name].update(value)

            if itr % self.adapt_cfg.log_freq == 0:
                log_string = " | ".join(
                    f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items()
                )
                elapsed = time.monotonic() - start_time
                LOGGER.info(
                    "[TRN] Iteration {:04d} | Elapsed: {} | Iterations/s: {:.4g} | {}".format(
                        itr,
                        readable_duration(elapsed),
                        self.adapt_cfg.log_freq / elapsed,
                        log_string,
                    )
                )

                loss_meters.clear()
                start_time = time.monotonic()

            if self.adapt_cfg.validate and itr % self.adapt_cfg.val_freq == 0:
                if itr == self.adapt_cfg.val_freq:  # first validation
                    baseline_metrics(self.cfg, datasets)
                log_metrics(self.cfg, model=self.encoder, data=datasets, step=itr)
                save_model(self.cfg, save_dir, model=self.encoder, itr=itr)

        LOGGER.info("Training has finished.")

        log_metrics(
            self.cfg,
            model=self.encoder,
            data=datasets,
            save_summary=True,
            step=itr,
            cluster_metrics=cluster_metrics,
        )
