"""Main training file"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
import logging
from pathlib import Path
import time
from typing import NamedTuple, cast

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
    AggregatorType,
    CmnistConfig,
    Config,
    DatasetConfig,
    DiscriminatorMethod,
    EncoderConfig,
    FdmConfig,
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
    count_parameters,
    flatten_dict,
    inf_generator,
    load_results,
    prod,
    random_seed,
    readable_duration,
    wandb_log,
)

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


class SemiSupervisedAlg(ExperimentBase, ABC):
    """Experiment singleton class."""

    def __init__(
        self,
        cfg: Config,
        data: DatasetConfig,
        misc: MiscConfig,
        device: torch.device,
    ) -> None:
        super().__init__(cfg=cfg, data=data, misc=misc, device=device)
        self.grad_scaler = GradScaler() if self.misc.use_amp else None

    def get_batch(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Batch]:
        x_c = self.to_device(next(context_data_itr)[0])
        x_c = cast(Tensor, x_c)
        x_t, s_t, y_t = self.to_device(*next(train_data_itr))
        return x_c, Batch(x_t, s_t, y_t)

    @abstractmethod
    def _train_step(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:
        ...


class AdvSemiSupervisedAlg(SemiSupervisedAlg, ABC):
    """Experiment singleton class."""

    def __init__(
        self,
        cfg: Config,
        data_cfg: DatasetConfig,
        gen_cfg: EncoderConfig,
        disc_cfg: FdmConfig,
        misc_cfg: MiscConfig,
        generator: AutoEncoder,
        adversary: Classifier | Discriminator,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor],
        predictor_y: Classifier,
        predictor_s: Classifier,
        device: torch.device,
    ) -> None:
        super().__init__(cfg=cfg, data=data_cfg, misc=misc_cfg, device=device)
        self.gen_cfg = gen_cfg
        self.disc_cfg = disc_cfg
        self.grad_scaler = GradScaler() if self.misc.use_amp else None
        self.generator = generator
        self.adversary = adversary
        self.recon_loss_fn = recon_loss_fn
        self.predictor_y = predictor_y
        self.predictor_s = predictor_s

    @implements(SemiSupervisedAlg)
    def _train_step(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:

        warmup = itr < self.disc_cfg.warmup_steps
        if (not warmup) and (self.disc_cfg.disc_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.disc_cfg.num_disc_updates):
                x_c, tr = self.get_batch(
                    context_data_itr=context_data_itr, train_data_itr=train_data_itr
                )
                self._update_adversary(x_c, tr.x)

        x_c, tr = self.get_batch(context_data_itr=context_data_itr, train_data_itr=train_data_itr)
        _, logging_dict = self._update(x_c=x_c, tr=tr, warmup=warmup)

        wandb_log(self.misc, logging_dict, step=itr)

        # Log images
        if itr % self.disc_cfg.log_freq == 0 and isinstance(self.data, ImageDatasetConfig):
            with torch.no_grad():
                self._log_recons(x=tr.x, itr=itr, prefix="train")
                self._log_recons(x=x_c, itr=itr, prefix="context")
        return logging_dict

    @abstractmethod
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        ...

    @abstractmethod
    def _update_adversary(self, x_c: Tensor, x_t: Tensor) -> tuple[Tensor, dict[str, float]]:
        ...

    @abstractmethod
    def _update(self, x_c: Tensor, tr: Batch, warmup: bool) -> tuple[Tensor, dict[str, float]]:
        ...


class SupportMatching(AdvSemiSupervisedAlg):
    # TODO: this is bad practice
    adversary: Discriminator

    @implements(AdvSemiSupervisedAlg)
    def _update_adversary(self, x_c: Tensor, x_t: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the generator constant.
        Args:
            x_c: x from the context set
            x_t: x from the training set
        """
        self.generator.eval()
        self.predictor_y.eval()
        self.predictor_s.eval()
        self.adversary.train()
        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):  # type: ignore
            encoding_t = self.generator.encode(x_t, stochastic=True)
            if not self.disc_cfg.train_on_recon:
                encoding_c = self.generator.encode(x_c, stochastic=True)

            if self.disc_cfg.train_on_recon:
                disc_input_c = x_c

            disc_loss = x_c.new_zeros(())
            logging_dict = {}
            disc_input_t = self._get_disc_input(encoding_t)
            disc_input_t = disc_input_t.detach()

            if not self.disc_cfg.train_on_recon:
                disc_input_c = self._get_disc_input(encoding_c)  # type: ignore
                disc_input_c = disc_input_c.detach()

            disc_loss = self.adversary.discriminator_loss(fake=disc_input_t, real=disc_input_c)  # type: ignore

            self.adversary.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            disc_loss = self.grad_scaler.scale(disc_loss)
        disc_loss.backward()

        self.adversary.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return disc_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _update(self, x_c: Tensor, tr: Batch, warmup: bool) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses.

        Args:
            x_t: x from the training set
        """
        # Compute losses for the generator.
        self.predictor_y.train()
        self.predictor_s.train()
        self.generator.train()
        self.adversary.eval()
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_t, gen_loss_tr, logging_dict_tr = self.generator.routine(
                tr.x, self.recon_loss_fn, self.disc_cfg.gen_weight
            )

            # ============================= recon loss for context set ============================
            encoding_c, gen_loss_ctx, logging_dict_ctx = self.generator.routine(
                x_c, self.recon_loss_fn, self.disc_cfg.gen_weight
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            gen_loss_tr = 0.5 * (gen_loss_tr + gen_loss_ctx)  # take average of the two recon losses
            gen_loss_tr *= self.disc_cfg.gen_loss_weight
            logging_dict["Loss Generator"] = gen_loss_tr
            total_loss = gen_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_t = self._get_disc_input(encoding_t)
                disc_input_c = self._get_disc_input(encoding_c)

                if self.disc_cfg.disc_method is DiscriminatorMethod.nn:
                    disc_loss = self.adversary.encoder_loss(
                        fake=disc_input_t, real=disc_input_c
                    )

                else:
                    x = disc_input_t
                    y = self._get_disc_input(encoding_c, detach=True)
                    disc_loss = mmd2(
                        x=x,
                        y=y,
                        kernel=self.disc_cfg.mmd_kernel,
                        scales=self.disc_cfg.mmd_scales,
                        wts=self.disc_cfg.mmd_wts,
                        add_dot=self.disc_cfg.mmd_add_dot,
                    )
                disc_loss *= self.disc_cfg.disc_weight
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.disc_cfg.pred_y_weight > 0:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_t.zy, tr.y)
                pred_y_loss *= self.disc_cfg.pred_y_weight
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.disc_cfg.pred_s_weight > 0:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_t.zs, tr.s)
                pred_s_loss *= self.disc_cfg.pred_s_weight
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self.generator.zero_grad()
        if self.disc_cfg.pred_y_weight > 0:
            self.predictor_y.zero_grad()
        if self.disc_cfg.pred_s_weight > 0:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            total_loss = self.grad_scaler.scale(total_loss)
        total_loss.backward()

        # Update the generator's parameters
        self.generator.step(grad_scaler=self.grad_scaler)
        if self.disc_cfg.pred_y_weight > 0:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.disc_cfg.pred_s_weight > 0:
            self.predictor_s.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return total_loss, logging_dict

    def _get_disc_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        if self.disc_cfg.train_on_recon:
            zs_m, _ = self.generator.mask(encoding, random=True, detach=detach)
            recon = self.generator.decode(zs_m, mode="relaxed")
            if self.gen_cfg.recon_loss is ReconstructionLoss.ce:
                recon = recon.argmax(dim=1).float() / 255
                if not isinstance(self.data, CmnistConfig):
                    recon = recon * 2 - 1
            return recon
        else:
            zs_m, _ = self.generator.mask(encoding, detach=detach)
            return self.generator.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""

        rows_per_block = 8
        num_blocks = 4
        if self.disc_cfg.aggregator_type is AggregatorType.none:
            num_sampled_bags = 0  # this is only defined here to make the linter happy
            num_samples = num_blocks * rows_per_block
        else:
            # take enough bags to have 32 samples
            num_sampled_bags = ((num_blocks * rows_per_block - 1) // self.disc_cfg.bag_size) + 1
            num_samples = num_sampled_bags * self.disc_cfg.bag_size

        sample = x[:num_samples]
        encoding = self.generator.encode(sample, stochastic=False)
        recon = self.generator.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.disc_cfg.train_on_recon:
            recons.append(recon.rand_s)
            caption += " | rand_s"

        to_log: list[Tensor] = [sample]
        for recon_ in recons:
            if self.gen_cfg.recon_loss is ReconstructionLoss.ce:
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

        if self.disc_cfg.aggregator_type is AggregatorType.gated:
            self.adversary(self._get_disc_input(encoding))
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
    def _update_disc(self, x_ctx: Tensor, tr_batch: Batch) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the generator constant.
        Args:
            x_c: x from the context set
            x_t: x from the training set
        """
        self.generator.eval()
        self.predictor_y.eval()
        self.predictor_s.eval()
        self.adversary.train()
        # Context-manager enables mixed-precision training
        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):  # type: ignore
            encoding_t = self.generator.encode(tr_batch.x, stochastic=True)
            disc_loss = x_ctx.new_zeros(())
            logging_dict = {}
            disc_input_t = self._get_disc_input(encoding_t)
            disc_input_t = disc_input_t.detach()
            disc_loss, _ = self.adversary.routine(encoding_t.zy, tr_batch.y)

        self.adversary.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            disc_loss = self.grad_scaler.scale(disc_loss)
        disc_loss.backward()

        self.adversary.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return disc_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _update(self, x_c: Tensor, tr: Batch, warmup: bool) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses.

        Args:
            x_t: x from the training set
        """
        # Compute losses for the generator.
        self.predictor_y.train()
        self.predictor_s.train()
        self.generator.train()
        self.adversary.eval()
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_t, gen_loss_tr, logging_dict_tr = self.generator.routine(
                tr.x, self.recon_loss_fn, self.disc_cfg.gen_weight
            )

            # ============================= recon loss for context set ============================
            # we need a reconstruction loss for x_c because...
            # ...when we train on encodings, the NN will otherwise just falsify encodings for x_c
            # ...when we train on recons, the GAN loss has it too easy to distinguish the two
            _, gen_loss_ctx, logging_dict_ctx = self.generator.routine(
                x_c, self.recon_loss_fn, self.disc_cfg.gen_weight
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            gen_loss_tr = 0.5 * (gen_loss_tr + gen_loss_ctx)  # take average of the two recon losses
            gen_loss_tr *= self.disc_cfg.gen_loss_weight
            logging_dict["Loss Generator"] = gen_loss_tr
            total_loss = gen_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_t = self._get_disc_input(encoding_t)
                disc_loss = self.adversary.routine(data=disc_input_t, targets=tr.y)[0]
                disc_loss *= self.disc_cfg.disc_weight
                # Negate the discriminator's loss to obtain the adversarial loss w.r.t the generator
                total_loss -= disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.disc_cfg.pred_y_weight > 0:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_t.zy, tr.y)
                pred_y_loss *= self.disc_cfg.pred_y_weight
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.disc_cfg.pred_s_weight > 0:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_t.zs, tr.s)
                pred_s_loss *= self.disc_cfg.pred_s_weight
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self.generator.zero_grad()
        if self.disc_cfg.pred_y_weight > 0:
            self.predictor_y.zero_grad()
        if self.disc_cfg.pred_s_weight > 0:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            total_loss = self.grad_scaler.scale(total_loss)
        total_loss.backward()

        # Update the generator's parameters
        self.generator.step(grad_scaler=self.grad_scaler)
        if self.disc_cfg.pred_y_weight > 0:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.disc_cfg.pred_s_weight > 0:
            self.predictor_s.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return total_loss, logging_dict

    def _get_disc_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        zs_m, _ = self.generator.mask(encoding, detach=detach)
        return self.generator.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""

        rows_per_block = 8
        num_blocks = 4
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = self.generator.encode(sample, stochastic=False)
        recon = self.generator.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]
        caption = "original | all | zero_s | just_s"
        to_log: list[Tensor] = [sample]
        for recon_ in recons:
            if self.gen_cfg.recon_loss is ReconstructionLoss.ce:
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
        the trained generator
    """
    # ==== initialize config shorthands ====
    data_cfg = cfg.data
    gen_cfg = cfg.gen
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

    save_dir = Path(to_absolute_path(misc_cfg.save_dir)) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        yaml.dump(as_pretty_dict(cfg), default_flow_style=False, allow_unicode=True, sort_keys=True)
    )
    LOGGER.info(f"Save directory: {save_dir.resolve()}")
    # ==== check GPU ====
    device = torch.device(misc_cfg.device)
    LOGGER.info(f"{torch.cuda.device_count()} GPUs available. Using device '{device}'")

    # ==== construct dataset ====
    datasets: DatasetTriplet = load_dataset(cfg)
    LOGGER.info(
        "Size of context-set: {}, training-set: {}, test-set: {}".format(
            len(datasets.context),  # type: ignore
            len(datasets.train),  # type: ignore
            len(datasets.test),  # type: ignore
        )
    )
    s_count = max(datasets.s_dim, 2)

    cluster_results = None
    cluster_test_metrics: dict[str, float] = {}
    cluster_context_metrics: dict[str, float] = {}
    if misc_cfg.cluster_label_file:
        cluster_results = load_results(cfg)
        cluster_test_metrics = cluster_results.test_metrics or {}
        cluster_context_metrics = cluster_results.context_metrics or {}
        context_sampler = get_stratified_sampler(
            group_ids=cluster_results.cluster_ids,
            oversample=disc_cfg.oversample,
            batch_size=disc_cfg.batch_size,
            min_size=None if disc_cfg.oversample else disc_cfg.eff_batch_size,
        )
        dataloader_kwargs = dict(sampler=context_sampler)
    elif disc_cfg.balanced_context:
        context_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.context,  # type: ignore
            s_count=s_count,
            batch_size=disc_cfg.eff_batch_size,
            oversample=disc_cfg.oversample,
            balance_hierarchical=False,
        )
        dataloader_kwargs = dict(sampler=context_sampler, shuffle=False)
    else:
        dataloader_kwargs = dict(shuffle=True)

    context_loader = DataLoader(
        datasets.context,
        batch_size=disc_cfg.eff_batch_size,
        num_workers=misc_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        **dataloader_kwargs,
    )

    train_sampler = build_weighted_sampler_from_dataset(
        dataset=datasets.train,  # type: ignore
        s_count=s_count,
        batch_size=disc_cfg.eff_batch_size,
        oversample=disc_cfg.oversample,
        balance_hierarchical=True,
    )
    train_loader = DataLoader(
        dataset=datasets.train,
        batch_size=disc_cfg.eff_batch_size,
        num_workers=misc_cfg.num_workers,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
    )
    context_data_itr = inf_generator(context_loader)
    train_data_itr = inf_generator(train_loader)
    # ==== construct networks ====
    input_shape = next(context_data_itr)[0][0].shape
    is_image_data = len(input_shape) > 2

    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    if is_image_data:
        decoding_dim = (
            input_shape[0] * 256 if gen_cfg.recon_loss is ReconstructionLoss.ce else input_shape[0]
        )
        decoder_out_act = None
        encoder, decoder, enc_dim = conv_autoencoder(
            input_shape,
            gen_cfg.init_chans,
            encoding_dim=gen_cfg.out_dim,
            decoding_dim=decoding_dim,
            levels=gen_cfg.levels,
            decoder_out_act=decoder_out_act,
            variational=disc_cfg.vae,
        )
    else:
        encoder, decoder, enc_dim = fc_autoencoder(
            input_shape,
            gen_cfg.init_chans,
            encoding_dim=gen_cfg.out_dim,
            levels=gen_cfg.levels,
            variational=disc_cfg.vae,
        )

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if gen_cfg.recon_loss is ReconstructionLoss.l1:
        recon_loss_fn = nn.L1Loss(reduction="sum")
    elif gen_cfg.recon_loss is ReconstructionLoss.l2:
        recon_loss_fn = nn.MSELoss(reduction="sum")
    elif gen_cfg.recon_loss is ReconstructionLoss.bce:
        recon_loss_fn = nn.BCELoss(reduction="sum")
    elif gen_cfg.recon_loss is ReconstructionLoss.huber:
        recon_loss_fn = lambda x, y: 0.1 * F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
    elif gen_cfg.recon_loss is ReconstructionLoss.ce:
        recon_loss_fn = PixelCrossEntropy(reduction="sum")
    elif gen_cfg.recon_loss is ReconstructionLoss.mixed:
        assert feature_group_slices is not None, "can only do multi gen_loss with feature groups"
        recon_loss_fn = MixedLoss(feature_group_slices, reduction="sum")
    else:
        raise ValueError(f"{gen_cfg.recon_loss} is an invalid reconstruction gen_loss")

    zs_dim = disc_cfg.zs_dim
    zy_dim = enc_dim - zs_dim
    encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)
    generator = build_ae(
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
        encoding_size=encoding_size,
        feature_group_slices=feature_group_slices,
    )
    # load pretrained encoder if one is provided
    if disc_cfg.use_pretrained_enc and cluster_results is not None:
        save_dict = torch.load(cluster_results.enc_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert args_encoder["encoder_type"] == "vae" if disc_cfg.vae else "ae"
            assert args_encoder["levels"] == gen_cfg.levels

    LOGGER.info(f"Encoding dim: {enc_dim}, {encoding_size}")

    # ================================== Initialise Discriminator =================================

    # TODO: Move into a 'build' method
    disc_optimizer_kwargs = {"lr": disc_cfg.disc_lr}
    disc_input_shape: tuple[int, ...] = input_shape if disc_cfg.train_on_recon else (enc_dim,)
    disc_fn: ModelFn
    if is_image_data and disc_cfg.train_on_recon:
        if isinstance(data_cfg, CmnistConfig):
            disc_fn = Strided28x28Net(batch_norm=False)
        else:
            disc_fn = Residual64x64Net(batch_norm=False)

    else:
        disc_fn = FcNet(hidden_dims=disc_cfg.disc_hidden_dims, activation=nn.GELU())
        # FcNet first flattens the input
        disc_input_shape = (
            (prod(disc_input_shape),)
            if isinstance(disc_input_shape, Sequence)
            else disc_input_shape
        )

    if disc_cfg.aggregator_type is not AggregatorType.none:
        final_proj = (
            FcNet(disc_cfg.aggregator_hidden_dims) if disc_cfg.aggregator_hidden_dims else None
        )
        aggregator: Aggregator
        if disc_cfg.aggregator_type is AggregatorType.kvq:
            aggregator = KvqAttentionAggregator(
                latent_dim=disc_cfg.aggregator_input_dim,
                bag_size=disc_cfg.bag_size,
                final_proj=final_proj,
                **disc_cfg.aggregator_kwargs,
            )
        else:
            aggregator = GatedAttentionAggregator(
                in_dim=disc_cfg.aggregator_input_dim,
                bag_size=disc_cfg.bag_size,
                final_proj=final_proj,
                **disc_cfg.aggregator_kwargs,
            )
        disc_fn = ModelAggregatorWrapper(
            disc_fn, aggregator, input_dim=disc_cfg.aggregator_input_dim
        )

    adversary = Discriminator(
        model=disc_fn(disc_input_shape, 1),  # type: ignore
        double_adv_loss=disc_cfg.double_adv_loss,
        optimizer_kwargs=disc_optimizer_kwargs,
        criterion=disc_cfg.disc_loss,
    )
    adversary.to(device)

    predictor_y = build_classifier(
        input_shape=(encoding_size.zy,),  # this is always trained on encodings
        target_dim=datasets.y_dim,
        model_fn=FcNet(hidden_dims=None),  # no hidden layers
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    predictor_y.to(device)

    predictor_s = build_classifier(
        input_shape=(encoding_size.zs,),  # this is always trained on encodings
        target_dim=datasets.s_dim,
        model_fn=FcNet(hidden_dims=None),  # no hidden layers
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    predictor_s.to(device)

    # TODO: allow switching between this and LAFTR
    exp = SupportMatching(
        disc_cfg=disc_cfg,
        cfg=cfg,
        data_cfg=data_cfg,
        gen_cfg=gen_cfg,
        misc_cfg=misc_cfg,
        generator=generator,
        adversary=adversary,
        recon_loss_fn=recon_loss_fn,
        predictor_y=predictor_y,
        predictor_s=predictor_s,
        device=device,
    )

    start_itr = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if misc_cfg.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        generator, start_itr = restore_model(cfg, Path(misc_cfg.resume), generator)
        generator = cast(AutoEncoder, generator)

        if misc_cfg.evaluate:
            log_metrics(
                cfg,
                generator,
                datasets,
                step=0,
                save_summary=True,
                cluster_test_metrics=cluster_test_metrics,
                cluster_context_metrics=cluster_context_metrics,
            )
            if run is not None:
                run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
            return generator

    if disc_cfg.snorm:

        def _snorm(_module: nn.Module) -> nn.Module:
            if isinstance(_module, nn.Conv2d):
                return torch.nn.utils.spectral_norm(_module)
            return _module

        adversary.apply(_snorm)  # type: ignore

    # Logging
    LOGGER.info(f"Number of trainable parameters: {count_parameters(generator)}")

    itr = start_itr
    adversary: nn.Module
    start_time = time.monotonic()

    loss_meters = defaultdict(AverageMeter)

    for itr in range(start_itr, disc_cfg.iters + 1):

        logging_dict = exp._train_step(
            context_data_itr=context_data_itr,
            train_data_itr=train_data_itr,
            itr=itr,
        )
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        if itr % disc_cfg.log_freq == 0:
            log_string = " | ".join(f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items())
            elapsed = time.monotonic() - start_time
            LOGGER.info(
                "[TRN] Iteration {:04d} | Elapsed: {} | Iterations/s: {:.4g} | {}".format(
                    itr,
                    readable_duration(elapsed),
                    disc_cfg.log_freq / elapsed,
                    log_string,
                )
            )

            loss_meters.clear()
            start_time = time.monotonic()

        if disc_cfg.validate and itr % disc_cfg.val_freq == 0:
            if itr == disc_cfg.val_freq:  # first validation
                baseline_metrics(cfg, datasets)
            log_metrics(cfg, model=generator, data=datasets, step=itr)
            save_model(cfg, save_dir, model=generator, itr=itr)

    LOGGER.info("Training has finished.")

    log_metrics(
        cfg,
        model=generator,
        data=datasets,
        save_summary=True,
        step=itr,
        cluster_test_metrics=cluster_test_metrics,
        cluster_context_metrics=cluster_context_metrics,
    )
    if run is not None:
        run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process

    return generator
