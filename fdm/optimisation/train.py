"""Main training file"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
import logging
from pathlib import Path
import time
from typing import NamedTuple, cast

import git
from hydra.utils import to_absolute_path
import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal
import wandb
import yaml

from fdm.models import AutoEncoder, Classifier, EncodingSize, build_discriminator
from fdm.models.configs import Residual64x64Net
from fdm.models.configs.classifiers import Strided28x28Net
from fdm.optimisation.mmd import mmd2
from shared.configs import (
    AggregatorType,
    Config,
    DatasetConfig,
    DiscriminatorMethod,
    EncoderConfig,
    FdmConfig,
    FdmDataset,
    MiscConfig,
    ReconstructionLoss,
)
from shared.data import DatasetTriplet, load_dataset
from shared.layers import Aggregator, GatedAttentionAggregator, KvqAttentionAggregator
from shared.models.configs import FcNet, ModelAggregatorWrapper, conv_autoencoder, fc_autoencoder
from shared.utils import (
    AverageMeter,
    ExperimentBase,
    ModelFn,
    StratifiedSampler,
    as_pretty_dict,
    class_id_to_label,
    count_parameters,
    flatten_dict,
    inf_generator,
    label_to_class_id,
    lcm,
    load_results,
    prod,
    random_seed,
    readable_duration,
    wandb_log,
)

from .build import build_ae
from .evaluation import baseline_metrics, log_metrics
from .loss import MixedLoss, PixelCrossEntropy
from .utils import get_stratified_sampler, log_images, restore_model, save_model

__all__ = ["main"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Triplet(NamedTuple):
    """A data structure for reducing clutter."""

    x: Tensor
    s: Tensor
    y: Tensor


class Experiment(ExperimentBase):
    """Experiment singleton class."""

    def __init__(
        self,
        args: FdmConfig,
        cfg: Config,
        data: DatasetConfig,
        enc: EncoderConfig,
        misc: MiscConfig,
        generator: AutoEncoder,
        disc_ensemble: nn.ModuleList,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor],
        predictor_y: Classifier,
        predictor_s: Classifier,
        device: torch.device,
    ) -> None:
        super().__init__(cfg=cfg, data=data, enc=enc, misc=misc, device=device)
        self.args = args
        self.grad_scaler = GradScaler() if self.misc.use_amp else None
        self.generator = generator
        self.disc_ensemble = disc_ensemble
        self.recon_loss_fn = recon_loss_fn
        self.predictor_y = predictor_y
        self.predictor_s = predictor_s

    def get_batch(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Triplet]:
        x_c = self.to_device(next(context_data_itr)[0])
        x_c = cast(Tensor, x_c)
        x_t, s_t, y_t = self.to_device(*next(train_data_itr))
        return x_c, Triplet(x_t, s_t, y_t)

    def train_step(
        self,
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        itr: int,
    ) -> dict[str, float]:

        warmup = itr < self.args.warmup_steps
        if (not warmup) and (self.args.disc_method is DiscriminatorMethod.nn):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.args.num_disc_updates):
                x_c, tr = self.get_batch(
                    context_data_itr=context_data_itr, train_data_itr=train_data_itr
                )
                self.update_disc(x_c, tr.x)

        x_c, tr = self.get_batch(context_data_itr=context_data_itr, train_data_itr=train_data_itr)
        _, logging_dict = self.update(x_c=x_c, tr=tr, warmup=warmup)

        wandb_log(self.misc, logging_dict, step=itr)

        # Log images
        if itr % self.args.log_freq == 0:
            with torch.no_grad():
                self.log_recons(x=tr.x, itr=itr)
        return logging_dict

    def update_disc(self, x_c: Tensor, x_t: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the generator constant.

        Args:
            x_c: x from the context set
            x_t: x from the training set
        """
        self.generator.eval()
        self.predictor_y.eval()
        self.predictor_s.eval()
        self.disc_ensemble.train()
        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):
            if self.args.aggregator_type is AggregatorType.none:
                ones = x_c.new_ones((x_c.size(0),))
                zeros = x_t.new_zeros((x_t.size(0),))
            else:
                ones = x_c.new_ones(self.args.batch_size)  # type: ignore
                zeros = x_c.new_zeros(self.args.batch_size)  # type: ignore

            if self.args.vae:
                encoding_t = self.generator.encode(x_t, stochastic=True)
                if not self.args.train_on_recon:
                    encoding_c = self.generator.encode(x_c, stochastic=True)
            else:
                encoding_t = self.generator.encode(x_t)
                if not self.args.train_on_recon:
                    encoding_c = self.generator.encode(x_c)

            if self.args.train_on_recon:
                disc_input_c = x_c

            disc_loss = x_c.new_zeros(())
            disc_acc = 0.0
            logging_dict = {}
            disc_input_t = self.get_disc_input(encoding_t)
            disc_input_t = disc_input_t.detach()

            if not self.args.train_on_recon:
                disc_input_c = self.get_disc_input(encoding_c)
                disc_input_c = disc_input_c.detach()

            for discriminator in self.disc_ensemble:
                discriminator = cast(Classifier, discriminator)
                disc_loss_true, acc_c = discriminator.routine(disc_input_c, ones)
                disc_loss_false, acc_t = discriminator.routine(disc_input_t, zeros)
                disc_loss += disc_loss_true + disc_loss_false
                disc_acc += 0.5 * (acc_c + acc_t)
            disc_loss /= len(self.disc_ensemble)

        logging_dict["Accuracy Discriminator (zy)"] = disc_acc / len(self.disc_ensemble)
        for discriminator in self.disc_ensemble:
            discriminator.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            disc_loss = self.grad_scaler.scale(disc_loss)
        disc_loss.backward()

        for discriminator in self.disc_ensemble:
            discriminator = cast(Classifier, discriminator)
            discriminator.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return disc_loss, logging_dict

    def update(self, x_c: Tensor, tr: Triplet, warmup: bool) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses.

        Args:
            x_t: x from the training set
        """
        # Compute losses for the generator.
        self.predictor_y.train()
        self.predictor_s.train()
        self.generator.train()
        self.disc_ensemble.eval()
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc.use_amp):
            # ============================= recon loss for training set ===========================
            encoding, elbo, logging_dict_elbo = self.generator.routine(
                tr.x, self.recon_loss_fn, self.args.kl_weight
            )

            # ============================= recon loss for context set ============================
            # we need a reconstruction loss for x_c because...
            # ...when we train on encodings, the NN will otherwise just falsify encodings for x_c
            # ...when we train on recons, the GAN loss has it too easy to distinguish the two
            encoding_c, elbo_c, logging_dict_elbo_c = self.generator.routine(
                x_c, self.recon_loss_fn, self.args.kl_weight
            )
            logging_dict.update(
                {k: v + logging_dict_elbo_c[k] for k, v in logging_dict_elbo.items()}
            )
            elbo = 0.5 * (elbo + elbo_c)  # take average of the two recon losses
            elbo *= self.args.elbo_weight
            logging_dict["ELBO"] = elbo
            total_loss = elbo

            # ================================= adversarial losses ================================
            disc_input_no_s = self.get_disc_input(encoding, invariant_to="s")

            if not warmup:
                if self.args.disc_method is DiscriminatorMethod.nn:
                    if self.args.aggregator_type is AggregatorType.none:
                        zeros = tr.x.new_zeros((tr.x.size(0),))
                    else:
                        zeros = x_c.new_zeros(self.args.batch_size)  # type: ignore

                    disc_loss = tr.x.new_zeros(())
                    disc_acc = 0.0
                    for discriminator in self.disc_ensemble:
                        discriminator = cast(Classifier, discriminator)
                        disc_loss_m, disc_acc_m = discriminator.routine(disc_input_no_s, zeros)
                        disc_loss += disc_loss_m
                        disc_acc += disc_acc_m
                    disc_loss /= len(self.disc_ensemble)
                    disc_acc /= len(self.disc_ensemble)
                    logging_dict["Accuracy Discriminator (zy)"] = disc_acc

                else:
                    x = disc_input_no_s
                    y = self.get_disc_input(encoding_c.detach(), invariant_to="s")
                    disc_loss = mmd2(
                        x=x,
                        y=y,
                        kernel=self.args.mmd_kernel,
                        scales=self.args.mmd_scales,
                        wts=self.args.mmd_wts,
                        add_dot=self.args.mmd_add_dot,
                    )
                disc_loss *= self.args.disc_weight
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            # this is a pretty cheap masking operation, so it's okay if it's not used
            enc_no_s, enc_no_y = self.generator.mask(encoding, random=False)
            if self.args.pred_y_weight > 0:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(enc_no_s, tr.y)
                pred_y_loss *= self.args.pred_y_weight
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.args.pred_s_weight > 0:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(enc_no_y, tr.s)
                pred_s_loss *= self.args.pred_s_weight
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self.generator.zero_grad()
        if self.args.pred_y_weight > 0:
            self.predictor_y.zero_grad()
        if self.args.pred_s_weight > 0:
            self.predictor_s.zero_grad()

        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            total_loss = self.grad_scaler.scale(total_loss)
        total_loss.backward()

        # Update the generator's parameters
        self.generator.step(grad_scaler=self.grad_scaler)
        if self.args.pred_y_weight > 0:
            self.predictor_y.step(grad_scaler=self.grad_scaler)
        if self.args.pred_s_weight > 0:
            self.predictor_s.step(grad_scaler=self.grad_scaler)
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

        return total_loss, logging_dict

    def get_disc_input(self, encoding: Tensor, invariant_to: Literal["s", "y"] = "s") -> Tensor:
        """Construct the input that the discriminator expects."""
        if self.args.train_on_recon:
            zs_m, zy_m = self.generator.mask(encoding, random=True)
            recon = self.generator.decode(zs_m if invariant_to == "s" else zy_m, mode="relaxed")
            if self.enc.recon_loss is ReconstructionLoss.ce:
                recon = recon.argmax(dim=1).float() / 255
                if self.data.dataset is not FdmDataset.cmnist:
                    recon = recon * 2 - 1
            return recon
        else:
            zs_m, zy_m = self.generator.mask(encoding)
            return zs_m if invariant_to == "s" else zy_m

    def log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""
        rows_per_block = 8
        num_blocks = 4
        sample = x[: num_blocks * rows_per_block]
        encoding = self.generator.encode(sample, stochastic=False)
        recon = self.generator.all_recons(encoding, mode="hard")

        to_log: tuple[Tensor, ...] = (sample, recon.all, recon.zero_s, recon.just_s)
        caption = "original | all | zero_s | just_s"
        if self.args.train_on_recon:
            to_log += (recon.rand_s,)
            caption += " | rand_s"
        ncols = len(to_log)

        interleaved = torch.stack(to_log, dim=1).view(
            ncols * num_blocks * rows_per_block, sample.size(1), sample.size(2), sample.size(3)
        )
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

        # log_images(self.cfg, x[:64], "original_x", step=itr, prefix=prefix)
        # log_images(self.cfg, recon.all, "reconstruction_all", step=itr, prefix=prefix)
        # log_images(self.cfg, recon.rand_s, "reconstruction_rand_s", step=itr, prefix=prefix)
        # log_images(self.cfg, recon.zero_s, "reconstruction_zero_s", step=itr, prefix=prefix)
        # log_images(self.cfg, recon.just_s, "reconstruction_just_s", step=itr, prefix=prefix)


def main(cfg: Config, cluster_label_file: Path | None = None) -> AutoEncoder:
    """Main function.

    Args:
        hydra_config: configuration object from hydra
        cluster_label_file: path to a pth file with cluster IDs

    Returns:
        the trained generator
    """
    # ==== initialize config shorthands ====
    args = cfg.fdm
    data = cfg.data
    enc = cfg.enc
    misc = cfg.misc

    assert args.test_batch_size  # test_batch_size defaults to eff_batch_size if unspecified
    # ==== current git commit ====
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    random_seed(misc.seed, misc.use_gpu)
    if cluster_label_file is not None:
        misc.cluster_label_file = str(cluster_label_file)

    run = None
    if misc.use_wandb:
        project_suffix = f"-{data.dataset.name}" if data.dataset is not FdmDataset.cmnist else ""
        group = ""
        if misc.log_method:
            group += misc.log_method
        if misc.exp_group:
            group += "." + misc.exp_group
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

    save_dir = Path(to_absolute_path(misc.save_dir)) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        yaml.dump(as_pretty_dict(cfg), default_flow_style=False, allow_unicode=True, sort_keys=True)
    )
    LOGGER.info(f"Save directory: {save_dir.resolve()}")
    # ==== check GPU ====
    device = torch.device(misc.device)
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
    if misc.cluster_label_file:
        cluster_results = load_results(cfg)
        cluster_test_metrics = cluster_results.test_metrics or {}
        cluster_context_metrics = cluster_results.context_metrics or {}
        context_sampler = get_stratified_sampler(
            group_ids=cluster_results.cluster_ids,
            oversample=args.oversample,
            batch_size=args.batch_size,
            min_size=None if args.oversample else args.eff_batch_size,
        )
        dataloader_kwargs = dict(sampler=context_sampler)
    elif args.balanced_context:
        context_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.context,
            s_count=s_count,
            test_batch_size=args.test_batch_size,
            batch_size=args.eff_batch_size,
            num_workers=0,  # can easily get stuck with more workers
            oversample=args.oversample,
            balance_hierarchical=False,
        )
        dataloader_kwargs = dict(sampler=context_sampler, shuffle=False)
    else:
        dataloader_kwargs = dict(shuffle=True)

    context_loader = DataLoader(
        datasets.context,
        batch_size=args.eff_batch_size,
        num_workers=misc.num_workers,
        pin_memory=True,
        drop_last=True,
        **dataloader_kwargs,
    )

    train_sampler = build_weighted_sampler_from_dataset(
        dataset=datasets.train,
        s_count=s_count,
        test_batch_size=args.test_batch_size,
        batch_size=args.eff_batch_size,
        num_workers=0,  # can easily get stuck with more workers
        oversample=args.oversample,
        balance_hierarchical=True,
    )
    train_loader = DataLoader(
        dataset=datasets.train,
        batch_size=args.eff_batch_size,
        num_workers=misc.num_workers,
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
            input_shape[0] * 256 if enc.recon_loss is ReconstructionLoss.ce else input_shape[0]
        )
        decoder_out_act = None
        encoder, decoder, enc_shape = conv_autoencoder(
            input_shape,
            enc.init_chans,
            encoding_dim=enc.out_dim,
            decoding_dim=decoding_dim,
            levels=enc.levels,
            decoder_out_act=decoder_out_act,
            variational=args.vae,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            input_shape,
            enc.init_chans,
            encoding_dim=enc.out_dim,
            levels=enc.levels,
            variational=args.vae,
        )

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if enc.recon_loss is ReconstructionLoss.l1:
        recon_loss_fn = nn.L1Loss(reduction="sum")
    elif enc.recon_loss is ReconstructionLoss.l2:
        recon_loss_fn = nn.MSELoss(reduction="sum")
    elif enc.recon_loss is ReconstructionLoss.bce:
        recon_loss_fn = nn.BCELoss(reduction="sum")
    elif enc.recon_loss is ReconstructionLoss.huber:
        recon_loss_fn = lambda x, y: 0.1 * F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
    elif enc.recon_loss is ReconstructionLoss.ce:
        recon_loss_fn = PixelCrossEntropy(reduction="sum")
    elif enc.recon_loss is ReconstructionLoss.mixed:
        assert feature_group_slices is not None, "can only do multi gen_loss with feature groups"
        recon_loss_fn = MixedLoss(feature_group_slices, reduction="sum")
    else:
        raise ValueError(f"{enc.recon_loss} is an invalid reconstruction gen_loss")

    zs_dim = round(args.zs_frac * enc_shape[0])
    zy_dim = enc_shape[0] - zs_dim
    encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)
    generator = build_ae(
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
        encoding_size=encoding_size,
        feature_group_slices=feature_group_slices,
    )
    # load pretrained encoder if one is provided
    if args.use_pretrained_enc and cluster_results is not None:
        save_dict = torch.load(cluster_results.enc_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert args_encoder["encoder_type"] == "vae" if args.vae else "ae"
            assert args_encoder["levels"] == enc.levels

    LOGGER.info(f"Encoding shape: {enc_shape}, {encoding_size}")

    # ================================== Initialise Discriminator =================================

    disc_optimizer_kwargs = {"lr": args.disc_lr}
    disc_input_shape: tuple[int, ...] = input_shape if args.train_on_recon else enc_shape
    disc_fn: ModelFn
    if is_image_data and args.train_on_recon:
        if data.dataset is FdmDataset.cmnist:
            disc_fn = Strided28x28Net(batch_norm=False)
        else:
            disc_fn = Residual64x64Net(batch_norm=False)

    else:
        disc_fn = FcNet(hidden_dims=args.disc_hidden_dims)
        # FcNet first flattens the input
        disc_input_shape = (
            (prod(disc_input_shape),)
            if isinstance(disc_input_shape, Sequence)
            else disc_input_shape
        )

    if args.aggregator_type is not AggregatorType.none:
        final_proj = FcNet(args.aggregator_hidden_dims) if args.aggregator_hidden_dims else None
        aggregator: Aggregator
        if args.aggregator_type is AggregatorType.kvq:
            aggregator = KvqAttentionAggregator(
                latent_dim=args.aggregator_input_dim,
                bag_size=args.bag_size,
                final_proj=final_proj,
                **args.aggregator_kwargs,
            )
        else:
            aggregator = GatedAttentionAggregator(
                in_dim=args.aggregator_input_dim,
                bag_size=args.bag_size,
                final_proj=final_proj,
                **args.aggregator_kwargs,
            )
        disc_fn = ModelAggregatorWrapper(disc_fn, aggregator, input_dim=args.aggregator_input_dim)

    disc_ensemble: nn.ModuleList = nn.ModuleList()
    for k in range(args.num_discs):
        disc = build_discriminator(
            input_shape=disc_input_shape,
            target_dim=1,  # real vs fake
            model_fn=disc_fn,
            optimizer_kwargs=disc_optimizer_kwargs,
        )
        disc_ensemble.append(disc)
    disc_ensemble.to(device)

    predictor_y = build_discriminator(
        input_shape=(prod(enc_shape),),  # this is always trained on encodings
        target_dim=datasets.y_dim,
        model_fn=FcNet(hidden_dims=None),  # no hidden layers
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    predictor_y.to(device)

    predictor_s = build_discriminator(
        input_shape=(prod(enc_shape),),  # this is always trained on encodings
        target_dim=datasets.s_dim,
        model_fn=FcNet(hidden_dims=None),  # no hidden layers
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    predictor_s.to(device)

    exp = Experiment(
        args=args,
        cfg=cfg,
        data=data,
        enc=enc,
        misc=misc,
        generator=generator,
        disc_ensemble=disc_ensemble,
        recon_loss_fn=recon_loss_fn,
        predictor_y=predictor_y,
        predictor_s=predictor_s,
        device=device,
    )

    start_itr = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if misc.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        generator, start_itr = restore_model(cfg, Path(misc.resume), generator)
        generator = cast(AutoEncoder, generator)

        if misc.evaluate:
            log_metrics(
                cfg,
                generator,
                datasets,
                0,
                save_to_csv=Path(to_absolute_path(misc.save_dir)),
                cluster_test_metrics=cluster_test_metrics,
                cluster_context_metrics=cluster_context_metrics,
            )
            if run is not None:
                run.join()  # this allows multiple experiments in one python process
            return generator

    if args.snorm:

        def _snorm(_module: nn.Module) -> nn.Module:
            if isinstance(_module, nn.Conv2d):
                return torch.nn.utils.spectral_norm(_module)
            return _module

        disc_ensemble.apply(_snorm)

    # Logging
    LOGGER.info(f"Number of trainable parameters: {count_parameters(generator)}")

    itr = start_itr
    disc: nn.Module
    start_time = time.monotonic()

    loss_meters = defaultdict(AverageMeter)

    for itr in range(start_itr, args.iters + 1):

        logging_dict = exp.train_step(
            context_data_itr=context_data_itr,
            train_data_itr=train_data_itr,
            itr=itr,
        )
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        if itr % args.log_freq == 0:
            log_string = " | ".join(f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items())
            elapsed = time.monotonic() - start_time
            LOGGER.info(
                "[TRN] Iteration {:04d} | Elapsed: {} | Iterations/s: {:.4g} | {}".format(
                    itr,
                    readable_duration(elapsed),
                    args.log_freq / elapsed,
                    log_string,
                )
            )

            loss_meters.clear()
            start_time = time.monotonic()

        if args.validate and itr % args.val_freq == 0:
            if itr == args.val_freq:  # first validation
                baseline_metrics(cfg, datasets, save_to_csv=Path(to_absolute_path(misc.save_dir)))
            log_metrics(cfg, model=generator, data=datasets, step=itr)
            save_model(cfg, save_dir, model=generator, itr=itr, sha=sha)

        if args.disc_reset_prob > 0:
            for k, discriminator in enumerate(disc_ensemble):
                discriminator = cast(Classifier, discriminator)
                if np.random.uniform() < args.disc_reset_prob:
                    LOGGER.info(f"Reinitializing discriminator {k}")
                    discriminator.reset_parameters()

    LOGGER.info("Training has finished.")

    log_metrics(
        cfg,
        model=generator,
        data=datasets,
        save_to_csv=Path(to_absolute_path(misc.save_dir)),
        step=itr,
        cluster_test_metrics=cluster_test_metrics,
        cluster_context_metrics=cluster_context_metrics,
    )
    if run is not None:
        run.join()  # this allows multiple experiments in one python process

    return generator


def build_weighted_sampler_from_dataset(
    dataset: Dataset,
    s_count: int,
    oversample: bool,
    test_batch_size: int,
    batch_size: int,
    num_workers: int,
    balance_hierarchical: bool,
) -> StratifiedSampler:
    #  Extract the s and y labels in a dataset-agnostic way (by iterating)
    data_loader = DataLoader(
        dataset=dataset, drop_last=False, batch_size=test_batch_size, num_workers=num_workers
    )
    s_all, y_all = [], []
    for _, s, y in data_loader:
        s_all.append(s)
        y_all.append(y)
    s_all = torch.cat(s_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    # Balance the batches of the training set via weighted sampling
    class_ids = label_to_class_id(s=s_all, y=y_all, s_count=s_count).view(-1)
    if balance_hierarchical:
        # Here we make sure that in a batch, y is balanced and within the y subsets, s is balanced.
        # So, if, for y=1, there are only samples with s=1, but for y=0, there's s=0 and s=1,
        # then the samples with y=1/s=1 get a multiplier of 2.
        multipliers, group_sizes = _get_multipliers_and_group_size(class_ids, s_count)
        return StratifiedSampler(
            group_ids=class_ids.tolist(),
            num_samples_per_group=max(group_sizes) if oversample else min(group_sizes),
            replacement=oversample,
            multipliers=multipliers,
        )
    else:
        return get_stratified_sampler(
            group_ids=class_ids, oversample=oversample, batch_size=batch_size
        )


def _get_multipliers_and_group_size(
    class_ids: Tensor, s_count: int
) -> tuple[dict[int, int], list[int]]:
    """This is a standalone function only because then we can have a unit test for it."""
    unique_classes, counts = torch.unique(class_ids, sorted=False, return_counts=True)
    class_ids_and_counts = [(int(i), int(c)) for i, c in zip(unique_classes, counts)]

    # first, count how many subgroups there are for each y
    num_subgroups_per_y = defaultdict(int)
    for class_id, count in class_ids_and_counts:
        corresponding_y = class_id_to_label(class_id, s_count, "y")
        num_subgroups_per_y[corresponding_y] += 1

    # To make all subgroups effectively the same size, we first scale everything by the least common
    # multiplier and then we divide by the number of subgroups to get the final multiplier.
    largest_multiplier = lcm(num_subgroups_per_y.values())
    multipliers = {}
    group_sizes = []
    for class_id, count in class_ids_and_counts:
        num_subgroups = num_subgroups_per_y[class_id_to_label(class_id, s_count, "y")]
        multiplier = largest_multiplier // num_subgroups
        multipliers[class_id] = multiplier
        group_sizes.append(count // multiplier)  # for book keeping, groups need to be smaller
    return multipliers, group_sizes
