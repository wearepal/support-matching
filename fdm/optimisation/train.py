"""Main training file"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
import logging
from pathlib import Path
import time
from typing import Literal, NamedTuple, cast

from hydra.utils import to_absolute_path
from kit import implements
import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import yaml

from fdm.models import AutoEncoder, Classifier, EncodingSize, build_classifier
from fdm.models.base import SplitEncoding
from fdm.models.configs import Residual64x64Net
from fdm.models.configs.classifiers import Strided28x28Net
from fdm.models.discriminator import Discriminator
from fdm.optimisation.mmd import mmd2
from shared.configs import (
    AdvConfig,
    AggregatorType,
    CmnistConfig,
    Config,
    DatasetConfig,
    DiscriminatorMethod,
    EncoderConfig,
    ImageDatasetConfig,
    MiscConfig,
    ReconstructionLoss,
)
from shared.data import DatasetTriplet, load_dataset
from shared.layers import Aggregator, GatedAttentionAggregator, KvqAttentionAggregator
from shared.models.configs import (
    FcNet,
    ModelAggregatorWrapper,
    conv_autoencoder,
    fc_autoencoder,
)
from shared.utils import (
    AverageMeter,
    ExperimentBase,
    ModelFn,
    as_pretty_dict,
    flatten_dict,
    inf_generator,
    load_results,
    prod,
    random_seed,
    readable_duration,
    wandb_log,
)
from shared.utils.loadsave import ClusterResults

from .build import build_ae
from .evaluation import baseline_metrics, log_metrics
from .loss import MixedLoss, PixelCrossEntropy
from .utils import (
    build_weighted_sampler_from_dataset,
    get_stratified_sampler,
    log_attention,
    log_images,
    restore_model,
    save_model,
)

__all__ = ["main"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Batch(NamedTuple):
    """A data structure for reducing clutter."""

    x: Tensor
    s: Tensor
    y: Tensor


class SemiSupervisedAlg(ExperimentBase):
    """Experiment singleton class."""

    def __init__(
        self,
        cfg: Config,
        data_cfg: DatasetConfig,
        misc_cfg: MiscConfig,
    ) -> None:
        super().__init__(cfg=cfg, data_cfg=data_cfg, misc_cfg=misc_cfg)
        self.grad_scaler = GradScaler() if self.misc_cfg.use_amp else None

    def sample_context(self, context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        return cast(Tensor, self.to_device(next(context_data_itr)[0]))

    def sample_train(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> Batch:
        x_tr, s_tr, y_tr = self.to_device(*next(train_data_itr))
        return Batch(x_tr, s_tr, y_tr)

    @abstractmethod
    def _train_step(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:
        ...


class AdvSemiSupervisedAlg(SemiSupervisedAlg):
    """Experiment singleton class."""

    _encoding_size: EncodingSize
    optimizer_kwargs: dict
    enerator: AutoEncoder
    adversary: Classifier | Discriminator
    recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
    predictor_y: Classifier
    predictor_s: Classifier

    def __init__(
        self,
        cfg: Config,
        data_cfg: DatasetConfig,
        enc_cfg: EncoderConfig,
        adv_cfg: AdvConfig,
        misc_cfg: MiscConfig,
    ) -> None:
        super().__init__(cfg=cfg, data_cfg=data_cfg, misc_cfg=misc_cfg)
        self.enc_cfg = enc_cfg
        self.adv_cfg = adv_cfg
        self.grad_scaler = GradScaler() if self.misc_cfg.use_amp else None

    def build_encoder(
        self,
        input_shape: tuple[int, ...],
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
                variational=self.adv_cfg.vae,
            )
        else:
            encoder, decoder, latent_dim = fc_autoencoder(
                input_shape,
                self.enc_cfg.init_chans,
                encoding_dim=self.enc_cfg.out_dim,
                levels=self.enc_cfg.levels,
                variational=self.adv_cfg.vae,
            )

        zs_dim = self.enc_cfg.zs_dim
        zy_dim = latent_dim - zs_dim
        self._encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)

        LOGGER.info(f"Encoding dim: {latent_dim}, {self._encoding_size}")

        encoder = build_ae(
            cfg=self.cfg,
            encoder=encoder,
            decoder=decoder,
            encoding_size=self._encoding_size,
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
    def build_adversary(self, input_shape: tuple[int, ...]) -> Classifier | Discriminator:
        ...

    def build_predictors(
        self, y_dim: int, s_dim: int
    ) -> tuple[Classifier | None, Classifier | None]:
        predictor_y = None
        if self.adv_cfg.pred_y_weight > 0:
            predictor_y = build_classifier(
                input_shape=(self._encoding_size.zy,),  # this is always trained on encodings
                target_dim=y_dim,
                model_fn=FcNet(hidden_dims=None),  # no hidden layers
                optimizer_kwargs=self.optimizer_kwargs["pred"],
            )
        predictor_s = None
        if self.adv_cfg.pred_s_weight > 0:
            predictor_s = build_classifier(
                input_shape=(self._encoding_size.zs,),  # this is always trained on encodings
                target_dim=s_dim,
                model_fn=FcNet(hidden_dims=None),  # no hidden layers
                optimizer_kwargs=self.optimizer_kwargs["pred"],
            )
        return predictor_y, predictor_s

    def build(
        self, input_shape: tuple[int, ...], y_dim: int, s_dim: int, feature_group_slices
    ) -> None:
        self.encoder = self.build_encoder(
            input_shape=input_shape, feature_group_slices=feature_group_slices
        )
        self.adversary = self.build_adversary(input_shape=input_shape)
        self.build_predictors(y_dim=y_dim, s_dim=s_dim)
        self.recon_loss_fn = self._get_recon_loss_fn(feature_group_slices=feature_group_slices)

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

    @implements(SemiSupervisedAlg)
    def _train_step(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:

        warmup = itr < self.adv_cfg.warmup_steps
        if (not warmup) and (self.adv_cfg.disc_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.adv_cfg.num_disc_updates):
                self._step_adversary(
                    train_data_itr=train_data_itr, context_data_itr=context_data_itr
                )

        batch_tr = self.sample_train(train_data_itr=train_data_itr)
        x_ctx = self.sample_context(context_data_itr=context_data_itr)
        _, logging_dict = self._step_encoder(x_c=x_ctx, tr=batch_tr, warmup=warmup)

        wandb_log(self.misc_cfg, logging_dict, step=itr)

        # Log images
        if itr % self.adv_cfg.log_freq == 0 and isinstance(self.data_cfg, ImageDatasetConfig):
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
        self, x_c: Tensor, tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        ...

    def _update_encoder(self, loss: Tensor) -> None:
        self.encoder.zero_grad()
        if self.adv_cfg.pred_y_weight > 0:
            self.predictor_y.zero_grad()
        if self.adv_cfg.pred_s_weight > 0:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            loss = self.grad_scaler.scale(loss)
        loss.backward()

        # Update the encoder's parameters
        self.encoder.step(grad_scaler=self.grad_scaler)
        if self.adv_cfg.pred_y_weight > 0:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.adv_cfg.pred_s_weight > 0:
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

    def train(self, mode: Literal["encoder", "adversary"]) -> None:
        if mode == "encoder":
            self.encoder.train()
            self.predictor_y.train()
            self.predictor_s.train()
            self.adversary.eval()
        else:
            self.encoder.eval()
            self.predictor_y.eval()
            self.predictor_s.eval()
            self.adversary.train()

    def _get_data_iterators(
        self, datasets: DatasetTriplet, cluster_results: ClusterResults | None = None
    ) -> tuple[Iterator[Batch], Iterator[Batch]]:

        s_count = max(datasets.s_dim, 2)
        if cluster_results is not None:
            context_sampler = get_stratified_sampler(
                group_ids=cluster_results.cluster_ids,
                oversample=self.adv_cfg.oversample,
                batch_size=self.adv_cfg.batch_size,
                min_size=None if self.adv_cfg.oversample else self.adv_cfg.eff_batch_size,
            )
            dataloader_kwargs = dict(sampler=context_sampler)
            if self.enc_cfg.use_pretrained_enc:
                self.enc_cfg.checkpoint_path = str(cluster_results.enc_path)
        elif self.adv_cfg.balanced_context:
            context_sampler = build_weighted_sampler_from_dataset(
                dataset=datasets.context,  # type: ignore
                s_count=s_count,
                batch_size=self.adv_cfg.eff_batch_size,
                oversample=self.adv_cfg.oversample,
                balance_hierarchical=False,
            )
            dataloader_kwargs = dict(sampler=context_sampler, shuffle=False)
        else:
            dataloader_kwargs = dict(shuffle=True)

        context_loader = DataLoader(
            datasets.context,
            batch_size=self.adv_cfg.eff_batch_size,
            num_workers=self.misc_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            **dataloader_kwargs,
        )

        train_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.train,  # type: ignore
            s_count=s_count,
            batch_size=self.adv_cfg.eff_batch_size,
            oversample=self.adv_cfg.oversample,
            balance_hierarchical=True,
        )
        train_loader = DataLoader(
            dataset=datasets.train,
            batch_size=self.adv_cfg.eff_batch_size,
            num_workers=self.misc_cfg.num_workers,
            drop_last=True,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=True,
        )
        context_data_itr = inf_generator(context_loader)
        train_data_itr = inf_generator(train_loader)

        return train_data_itr, context_data_itr

    def fit(self, datasets: DatasetTriplet) -> None:
        # Load cluster results
        cluster_results = None
        cluster_test_metrics: dict[str, float] = {}
        cluster_context_metrics: dict[str, float] = {}
        if self.misc_cfg.cluster_label_file:
            cluster_results = load_results(self.cfg)
            cluster_test_metrics = cluster_results.test_metrics or {}
            cluster_context_metrics = cluster_results.context_metrics or {}

        # Construct the data iterators
        train_data_itr, context_data_itr = self._get_data_iterators(
            datasets=datasets, cluster_results=cluster_results
        )
        # ==== construct networks ====
        input_shape = next(context_data_itr)[0][0].shape
        feature_group_slices = getattr(datasets.context, "feature_group_slices", None)
        self.build(
            input_shape=input_shape,
            y_dim=datasets.y_dim,
            s_dim=datasets.s_dim,
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
                    cluster_test_metrics=cluster_test_metrics,
                    cluster_context_metrics=cluster_context_metrics,
                )

        itr = start_itr
        start_time = time.monotonic()
        loss_meters = defaultdict(AverageMeter)

        for itr in range(start_itr, self.adv_cfg.iters + 1):

            logging_dict = self._train_step(
                context_data_itr=context_data_itr,
                train_data_itr=train_data_itr,
                itr=itr,
            )
            for name, value in logging_dict.items():
                loss_meters[name].update(value)

            if itr % self.adv_cfg.log_freq == 0:
                log_string = " | ".join(
                    f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items()
                )
                elapsed = time.monotonic() - start_time
                LOGGER.info(
                    "[TRN] Iteration {:04d} | Elapsed: {} | Iterations/s: {:.4g} | {}".format(
                        itr,
                        readable_duration(elapsed),
                        self.adv_cfg.log_freq / elapsed,
                        log_string,
                    )
                )

                loss_meters.clear()
                start_time = time.monotonic()

            if self.adv_cfg.validate and itr % self.adv_cfg.val_freq == 0:
                if itr == self.adv_cfg.val_freq:  # first validation
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
            cluster_test_metrics=cluster_test_metrics,
            cluster_context_metrics=cluster_context_metrics,
        )


class SupportMatching(AdvSemiSupervisedAlg):
    # TODO: this is bad practice
    adversary: Discriminator

    @implements(AdvSemiSupervisedAlg)
    def _build_adversary(self, input_shape: tuple[int, ...]) -> Discriminator:
        # TODO: Move into a 'build' method

        disc_optimizer_kwargs = {"lr": self.adv_cfg.disc_lr}
        disc_input_shape: tuple[int, ...] = (
            input_shape if self.adv_cfg.train_on_recon else (self.enc_cfg.out_dim,)
        )
        disc_fn: ModelFn
        if len(input_shape) > 2 and self.adv_cfg.train_on_recon:
            if isinstance(self.data_cfg, CmnistConfig):
                disc_fn = Strided28x28Net(batch_norm=False)
            else:
                disc_fn = Residual64x64Net(batch_norm=False)

        else:
            disc_fn = FcNet(hidden_dims=self.adv_cfg.disc_hidden_dims, activation=nn.GELU())
            # FcNet first flattens the input
            disc_input_shape = (
                (prod(disc_input_shape),)
                if isinstance(disc_input_shape, Sequence)
                else disc_input_shape
            )

        if self.adv_cfg.aggregator_type is not AggregatorType.none:
            final_proj = (
                FcNet(self.adv_cfg.aggregator_hidden_dims)
                if self.adv_cfg.aggregator_hidden_dims
                else None
            )
            aggregator: Aggregator
            if self.adv_cfg.aggregator_type is AggregatorType.kvq:
                aggregator = KvqAttentionAggregator(
                    latent_dim=self.adv_cfg.aggregator_input_dim,
                    bag_size=self.adv_cfg.bag_size,
                    final_proj=final_proj,
                    **self.adv_cfg.aggregator_kwargs,
                )
            else:
                aggregator = GatedAttentionAggregator(
                    in_dim=self.adv_cfg.aggregator_input_dim,
                    bag_size=self.adv_cfg.bag_size,
                    final_proj=final_proj,
                    **self.adv_cfg.aggregator_kwargs,
                )
            disc_fn = ModelAggregatorWrapper(
                disc_fn, aggregator, input_dim=self.adv_cfg.aggregator_input_dim
            )

        return Discriminator(
            model=disc_fn(disc_input_shape, 1),  # type: ignore
            double_adv_loss=self.adv_cfg.double_adv_loss,
            optimizer_kwargs=disc_optimizer_kwargs,
            criterion=self.adv_cfg.disc_loss,
        )

    @implements(AdvSemiSupervisedAlg)
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder constant.
        Args:
            x_c: x from the context set
            x_t: x from the training set

        """
        self.train("adversary")
        x_tr = self.sample_train(train_data_itr).x
        x_ctx = self.sample_context(context_data_itr=context_data_itr)

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            encoding_tr = self.encoder.encode(x_tr, stochastic=True)
            if not self.adv_cfg.train_on_recon:
                encoding_ctx = self.encoder.encode(x_ctx, stochastic=True)

            if self.adv_cfg.train_on_recon:
                adv_input_ctx = x_ctx

            adv_loss = x_ctx.new_zeros(())
            logging_dict = {}
            adv_input_tr = self._get_adv_input(encoding_tr)
            adv_input_tr = adv_input_tr.detach()

            if not self.adv_cfg.train_on_recon:
                with torch.no_grad():
                    adv_input_ctx = self._get_adv_input(encoding_ctx)  # type: ignore

            adv_loss = self.adversary.discriminator_loss(fake=adv_input_tr, real=adv_input_ctx)  # type: ignore

        self._update_adversary(adv_loss)

        return adv_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_ctx: Tensor, batch_tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses.

        Args:
            x_t: x from the training set
        """
        # Compute losses for the encoder.
        self.train("encoder")
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_t, enc_loss_tr, logging_dict_tr = self.encoder.routine(
                batch_tr.x, self.recon_loss_fn, self.adv_cfg.enc_loss_w
            )

            # ============================= recon loss for context set ============================
            encoding_c, enc_loss_ctx, logging_dict_ctx = self.encoder.routine(
                x_ctx, self.recon_loss_fn, self.adv_cfg.enc_loss_w
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_ctx)  # take average of the two recon losses
            enc_loss_tr *= self.adv_cfg.enc_loss_w
            logging_dict["Loss Generator"] = enc_loss_tr
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_t = self._get_adv_input(encoding_t)
                disc_input_c = self._get_adv_input(encoding_c)

                if self.adv_cfg.disc_method is DiscriminatorMethod.nn:
                    disc_loss = self.adversary.encoder_loss(fake=disc_input_t, real=disc_input_c)

                else:
                    x = disc_input_t
                    y = self._get_adv_input(encoding_c, detach=True)
                    disc_loss = mmd2(
                        x=x,
                        y=y,
                        kernel=self.adv_cfg.mmd_kernel,
                        scales=self.adv_cfg.mmd_scales,
                        wts=self.adv_cfg.mmd_wts,
                        add_dot=self.adv_cfg.mmd_add_dot,
                    )
                disc_loss *= self.adv_cfg.disc_weight
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.adv_cfg.pred_y_weight > 0:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_t.zy, batch_tr.y)
                pred_y_loss *= self.adv_cfg.pred_y_weight
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.adv_cfg.pred_s_weight > 0:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_t.zs, batch_tr.s)
                pred_s_loss *= self.adv_cfg.pred_s_weight
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self._update_encoder(total_loss)

        return total_loss, logging_dict

    def _get_adv_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        if self.adv_cfg.train_on_recon:
            zs_m, _ = self.encoder.mask(encoding, random=True, detach=detach)
            recon = self.encoder.decode(zs_m, mode="relaxed")
            if self.enc_cfg.recon_loss is ReconstructionLoss.ce:
                recon = recon.argmax(dim=1).float() / 255
                if not isinstance(self.data_cfg, CmnistConfig):
                    recon = recon * 2 - 1
            return recon
        else:
            zs_m, _ = self.encoder.mask(encoding, detach=detach)
            return self.encoder.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""

        rows_per_block = 8
        num_blocks = 4
        if self.adv_cfg.aggregator_type is AggregatorType.none:
            num_sampled_bags = 0  # this is only defined here to make the linter happy
            num_samples = num_blocks * rows_per_block
        else:
            # take enough bags to have 32 samples
            num_sampled_bags = ((num_blocks * rows_per_block - 1) // self.adv_cfg.bag_size) + 1
            num_samples = num_sampled_bags * self.adv_cfg.bag_size

        sample = x[:num_samples]
        encoding = self.encoder.encode(sample, stochastic=False)
        recon = self.encoder.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.adv_cfg.train_on_recon:
            recons.append(recon.rand_s)
            caption += " | rand_s"

        to_log: list[Tensor] = [sample]
        for recon_ in recons:
            if self.enc_cfg.recon_loss is ReconstructionLoss.ce:
                to_log.append(recon_.argmax(dim=1).float() / 255)
            else:
                to_log.append(recon_)
        ncols = len(to_log)

        interleaved = torch.stack(to_log, dim=1).view(ncols * num_samples, *sample.shape[1:])

        log_images(
            self.cfg,
            interleaved,
            name="reconstructions",
            step=itr,
            nsamples=[ncols * rows_per_block] * num_blocks,
            ncols=ncols,
            prefix=prefix,
            caption=caption,
        )

        if self.adv_cfg.aggregator_type is AggregatorType.gated:
            self.adversary(self._get_adv_input(encoding))
            assert isinstance(self.adversary.model[-1], Aggregator)  # type: ignore
            attention_weights = self.adversary.model[-1].attention_weights  # type: ignore
        log_attention(
            self.cfg,
            images=sample,
            attention_weights=attention_weights,  # type: ignore
            name="attention Weights",
            step=itr,
            nbags=num_sampled_bags,
            ncols=ncols,
            prefix=prefix,
        )


class LAFTR(AdvSemiSupervisedAlg):
    adversary: Classifier

    @implements(AdvSemiSupervisedAlg)
    def _build_adversary(self, input_shape: tuple[int, ...]) -> Classifier:
        # TODO: Move into a 'build' method

        adv_opt_kwaegs = {"lr": self.adv_cfg.disc_lr}
        adv_input_shape: tuple[int, ...] = (
            input_shape if self.adv_cfg.train_on_recon else (self.enc_cfg.out_dim,)
        )
        adv_fn: ModelFn
        if len(input_shape) > 2 and self.adv_cfg.train_on_recon:
            if isinstance(self.data_cfg, CmnistConfig):
                adv_fn = Strided28x28Net(batch_norm=False)
            else:
                adv_fn = Residual64x64Net(batch_norm=False)

        else:
            adv_fn = FcNet(hidden_dims=self.adv_cfg.disc_hidden_dims, activation=nn.GELU())
            # FcNet first flattens the input
            adv_input_shape = (
                (prod(adv_input_shape),)
                if isinstance(adv_input_shape, Sequence)
                else adv_input_shape
            )

        return Discriminator(
            model=adv_fn(adv_input_shape, self.predictor_s.num_classes),  # type: ignore
            double_adv_loss=self.adv_cfg.double_adv_loss,
            optimizer_kwargs=adv_opt_kwaegs,
            criterion=self.adv_cfg.disc_loss,
        )

    @implements(AdvSemiSupervisedAlg)
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder constant.
        Args:
            x_c: x from the context set
            x_t: x from the training set
        """
        self.train("adversary")
        tr_batch = self.sample_train(train_data_itr)

        logging_dict = {}
        # Context-manager enables mixed-precision training
        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            with torch.no_grad():
                encoding_tr = self.encoder.encode(tr_batch.x, stochastic=True)
                adv_input_tr = self._get_adv_input(encoding_tr)
            adv_loss, _ = self.adversary.routine(adv_input_tr, tr_batch.s)

        self._update_adversary(adv_loss)
        return adv_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_c: Tensor, tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses.

        Args:
            x_t: x from the training set
        """
        # Compute losses for the encoder.
        self.train("encoder")
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_t, enc_loss_tr, logging_dict_tr = self.encoder.routine(
                tr.x, self.recon_loss_fn, self.adv_cfg.enc_loss_w
            )

            # ============================= recon loss for context set ============================
            _, enc_loss_ctx, logging_dict_ctx = self.encoder.routine(
                x_c, self.recon_loss_fn, self.adv_cfg.enc_loss_w
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_ctx)  # take average of the two recon losses
            enc_loss_tr *= self.adv_cfg.enc_loss_w
            logging_dict["Loss Generator"] = enc_loss_tr
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_t = self._get_adv_input(encoding_t)
                disc_loss = self.adversary.routine(data=disc_input_t, targets=tr.y)[0]
                disc_loss *= self.adv_cfg.disc_weight
                # Negate the discriminator's loss to obtain the adversarial loss w.r.t the encoder
                total_loss -= disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.adv_cfg.pred_y_weight > 0:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_t.zy, tr.y)
                pred_y_loss *= self.adv_cfg.pred_y_weight
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.adv_cfg.pred_s_weight > 0:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_t.zs, tr.s)
                pred_s_loss *= self.adv_cfg.pred_s_weight
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self._update_encoder(total_loss)

        return total_loss, logging_dict

    def _get_adv_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        zs_m, _ = self.encoder.mask(encoding, detach=detach)
        return self.encoder.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""

        rows_per_block = 8
        num_blocks = 4
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = self.encoder.encode(sample, stochastic=False)
        recon = self.encoder.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]
        caption = "original | all | zero_s | just_s"
        to_log: list[Tensor] = [sample]
        for recon_ in recons:
            if self.enc_cfg.recon_loss is ReconstructionLoss.ce:
                to_log.append(recon_.argmax(dim=1).float() / 255)
            else:
                to_log.append(recon_)
        ncols = len(to_log)

        interleaved = torch.stack(to_log, dim=1).view(ncols * num_samples, *sample.shape[1:])

        log_images(
            self.cfg,
            interleaved,
            name="reconstructions",
            step=itr,
            nsamples=[ncols * rows_per_block] * num_blocks,
            ncols=ncols,
            prefix=prefix,
            caption=caption,
        )


def main(cfg: Config, cluster_label_file: Path | None = None) -> AutoEncoder:
    """Main function.

    Args:
        hydra_config: configuration object from hydra
        cluster_label_file: path to a pth file with cluster IDs

    Returns:
        the trained encoder
    """
    # ==== initialize config shorthands ====
    data_cfg = cfg.data
    enc_cfg = cfg.enc
    disc_cfg = cfg.disc
    misc_cfg = cfg.misc

    assert disc_cfg.test_batch_size  # test_batch_size defaults to eff_batch_size if unspecified

    random_seed(misc_cfg.seed, misc_cfg.use_gpu)
    if cluster_label_file is not None:
        misc_cfg.cluster_label_file = str(cluster_label_file)

    run = None
    if misc_cfg.use_wandb:
        project_suffix = f"-{data_cfg.log_name}" if not isinstance(data_cfg, CmnistConfig) else ""
        group = ""
        if misc_cfg.log_method:
            group += misc_cfg.log_method
        if misc_cfg.exp_group:
            group += "." + misc_cfg.exp_group
        if cfg.bias.log_dataset:
            group += "." + cfg.bias.log_dataset
        local_dir = Path(".", "local_logging")
        local_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity="predictive-analytics-lab",
            project="fdm-hydra" + project_suffix,
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(cfg)),
            group=group if group else None,
            reinit=True,
        )
        run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

    LOGGER.info(
        yaml.dump(as_pretty_dict(cfg), default_flow_style=False, allow_unicode=True, sort_keys=True)
    )

    # ==== construct dataset ====
    datasets: DatasetTriplet = load_dataset(cfg)
    LOGGER.info(
        "Size of context-set: {}, training-set: {}, test-set: {}".format(
            len(datasets.context),  # type: ignore
            len(datasets.train),  # type: ignore
            len(datasets.test),  # type: ignore
        )
    )

    # TODO: allow switching between this and LAFTR
    alg = SupportMatching(
        adv_cfg=disc_cfg,
        cfg=cfg,
        data_cfg=data_cfg,
        enc_cfg=enc_cfg,
        misc_cfg=misc_cfg,
    )
    alg.fit(datasets=datasets)

    if run is not None:
        run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process

    return alg.encoder
