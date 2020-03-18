"""Main training file"""
import time
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, cast
from itertools import islice

import git
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

import wandb
from fdm.configs import VaeArgs
from fdm.data import DatasetTriplet, load_dataset
from fdm.models import VAE, VaeResults, build_discriminator, AutoEncoder, Classifier
from fdm.models.configs import conv_autoencoder, fc_autoencoder, linear_disciminator
from fdm.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    random_seed,
    readable_duration,
    wandb_log,
    inf_generator,
)

from .evaluation import log_metrics
from .loss import PixelCrossEntropy, VGGLoss, grad_reverse, MixedLoss
from .utils import get_data_dim, log_images, save_model, restore_model

__all__ = ["main"]

NDECS = 0
ARGS: VaeArgs = None
LOGGER: Logger = None
INPUT_SHAPE: Tuple[int, ...] = ()


def compute_loss(
    x_p: Tensor, x_t: Tensor, vae: AutoEncoder, discriminator: Classifier, recon_loss_fn
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute all losses.

    Args:
        x_p: x from pre-training set
        x_t: x from training set
    """
    # Encode the data
    if ARGS.stochastic:
        vae = cast(VAE, vae)
        encoding, posterior = vae.encode(x_t, stochastic=True, return_posterior=True)
        kl_div = vae.compute_divergence(encoding, posterior)
    else:
        encoding = vae.encode(x_t)
        kl_div = x_t.new_zeros(())

    zs_m, zy_m = vae.random_mask(encoding)

    recon_all = vae.decode(encoding)
    recon_rand_s = grad_reverse(vae.decode(zs_m))
    recon_rand_y = grad_reverse(vae.decode(zy_m))

    # Compute losses
    recon_loss = recon_loss_fn(recon_all, x_t)

    recon_loss /= x_t.size(0)
    kl_div /= x_t.size(0)

    elbo = recon_loss + ARGS.kl_weight * kl_div

    # Discriminator for zy
    x_t_batch = (x_t.size(0),)
    disc_loss_rand_s, disc_acc_s = discriminator.routine(recon_rand_s, x_t.new_zeros(x_t_batch))
    disc_loss_rand_y, disc_acc_y = discriminator.routine(recon_rand_y, x_t.new_zeros(x_t_batch))

    disc_loss_true, disc_acc_true = discriminator.routine(x_p, x_p.new_ones((x_p.size(0),)))

    disc_loss = disc_loss_rand_s + disc_loss_rand_y + disc_loss_true

    elbo *= ARGS.elbo_weight
    disc_loss *= ARGS.pred_s_weight

    loss = elbo + disc_loss
    logging_dict = {
        "ELBO": elbo.item(),
        "Loss Adversarial": disc_loss.item(),
        "Accuracy Disc (rand s)": disc_acc_s,
        "Accuracy Disc (rand y)": disc_acc_y,
        "Accuracy Disc (true)": disc_acc_true,
        "KL divergence": kl_div.item(),
        "Loss Reconstruction": recon_loss.item(),
        "Loss Validation": (elbo - disc_loss).item(),
    }
    return loss, logging_dict


def train(
    vae: AutoEncoder,
    discriminator: Classifier,
    pretrain_data: DataLoader,
    task_train_data: DataLoader,
    epoch: int,
    recon_loss_fn,
) -> int:
    vae.train()

    total_loss_meter = AverageMeter()
    loss_meters: Optional[Dict[str, AverageMeter]] = None

    time_meter = AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    epoch_len = max(len(pretrain_data), len(task_train_data))
    itr = start_itr = (epoch - 1) * epoch_len
    data_iterator = islice(
        zip(inf_generator(pretrain_data), inf_generator(task_train_data)), epoch_len
    )

    for itr, ((x_p, _, _), (x_t, _, _)) in enumerate(data_iterator, start=start_itr):

        x_p, x_t = to_device(x_p, x_t)

        loss, logging_dict = compute_loss(
            x_p=x_p,
            x_t=x_t,
            vae=vae,
            discriminator=discriminator,
            recon_loss_fn=recon_loss_fn,
        )

        vae.zero_grad()
        discriminator.zero_grad()

        loss.backward()
        vae.step()
        if itr & ARGS.skip_disc_steps == 0:
            discriminator.step()

        # Log losses
        total_loss_meter.update(loss.item())
        if loss_meters is None:
            loss_meters = {name: AverageMeter() for name in logging_dict}
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        time_for_batch = time.time() - end
        time_meter.update(time_for_batch)

        wandb_log(ARGS, logging_dict, step=itr)
        end = time.time()

        # Log images
        if itr % ARGS.log_freq == 0:
            with torch.set_grad_enabled(False):
                log_recons(vae=vae, x_t=x_t, itr=itr)

    time_for_epoch = time.time() - start_epoch_time
    assert loss_meters is not None
    log_string = " | ".join(f"{name}: {meter.avg:.5g}" for name, meter in loss_meters.items())
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {} | Batches/s: {:.4g} | {} ({:.5g})",
        epoch,
        readable_duration(time_for_epoch),
        1 / time_meter.avg,
        log_string,
        total_loss_meter.avg,
    )
    return itr


def validate(vae: AutoEncoder, discriminator, val_loader, itr: int, recon_loss_fn):
    vae.eval()
    with torch.set_grad_enabled(False):
        loss_meter = AverageMeter()
        for val_itr, (x_val, _, _) in enumerate(val_loader):

            x_val = to_device(x_val)

            _, logging_dict = compute_loss(
                x_p=x_val,
                x_t=x_val,
                vae=vae,
                discriminator=discriminator,
                recon_loss_fn=recon_loss_fn,
            )

            loss_meter.update(logging_dict["Loss Validation"], n=x_val.size(0))

            if val_itr == 0:
                if ARGS.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):
                    log_recons(vae, x_val, itr=itr, prefix="test")
                else:
                    # TODO: the following is (most likely) completely wrong
                    z = vae(x_val[:1000])
                    _, recon_y, recon_s = vae.decode(z)
                    log_images(ARGS, x_val, "original_x", prefix="test", step=itr)
                    log_images(ARGS, recon_y, "reconstruction_yn", prefix="test", step=itr)
                    log_images(ARGS, recon_s, "reconstruction_yn", prefix="test", step=itr)
                    x_recon = vae(vae(x_val), reverse=True)
                    x_diff = (x_recon - x_val).abs().mean().item()
                    print(f"MAE of x and reconstructed x: {x_diff}")
                    wandb_log(ARGS, {"reconstruction MAE": x_diff}, step=itr)

        wandb_log(ARGS, {"Loss": loss_meter.avg}, step=itr)

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS._device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def log_recons(vae: AutoEncoder, x_t: Tensor, itr, prefix=None):
    """Log reconstructed images"""
    encoding = vae.encode(x_t[:64])

    # TODO: turn the following 4 lines into a single function; it's done quite often
    # (it corresponds to decode(..., partials=True))
    zs_m, zy_m = vae.random_mask(encoding)
    recon_all = vae.decode(encoding)
    recon_y = vae.decode(zs_m)
    recon_s = vae.decode(zy_m)

    log_images(ARGS, x_t[:64], "original_x", step=itr, prefix=prefix)
    log_images(ARGS, recon_all, "reconstruction_all", step=itr, prefix=prefix)
    log_images(ARGS, recon_y, "reconstruction_y", step=itr, prefix=prefix)
    log_images(ARGS, recon_s, "reconstruction_s", step=itr, prefix=prefix)
    # log_images(ARGS, recon_null, "reconstruction_null", step=itr, prefix=prefix)


def main(raw_args: Optional[List[str]] = None) -> AutoEncoder:
    """Main function

    Args:
        raw_args: commandline arguments
        datasets: a Dataset object

    Returns:
        the trained model
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    args = VaeArgs(explicit_bool=True, underscores_to_dashes=True)
    args.parse_args(raw_args)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER, INPUT_SHAPE
    ARGS = args
    args_dict = args.as_dict()

    if ARGS.use_wandb:
        wandb.init(project="fdm", config=args_dict)

    save_dir = Path(ARGS.save_dir) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info("Namespace(" + ", ".join(f"{k}={args_dict[k]}" for k in sorted(args_dict)) + ")")
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS._device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and ARGS.gpu >= 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using device '{}'", torch.cuda.device_count(), ARGS._device)

    # ==== construct dataset ====
    LOGGER.info(
        "Size of pretrain: {}, task_train: {}, task: {}",
        len(datasets.pretrain),
        len(datasets.task_train),
        len(datasets.task),
    )
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(
        datasets.pretrain,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.task_train,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

    # ==== construct networks ====
    INPUT_SHAPE = get_data_dim(train_loader)
    is_image_data = len(INPUT_SHAPE) > 2

    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    feature_group_slices = getattr(datasets.pretrain, "feature_group_slices", None)

    ARGS._s_dim = ARGS._s_dim if ARGS._s_dim > 1 else 2

    if is_image_data:
        decoding_dim = INPUT_SHAPE[0] * 256 if args.recon_loss == "ce" else INPUT_SHAPE[0]
        encoder, decoder, enc_shape = conv_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_y_dim + ARGS.enc_s_dim,
            decoding_dim=decoding_dim,
            levels=ARGS.levels,
            vae=ARGS.vae,
            s_dim=ARGS._s_dim if ARGS.cond_decoder else 0,
            level_depth=ARGS.level_depth,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_y_dim + ARGS.enc_s_dim,
            levels=ARGS.levels,
            vae=ARGS.vae,
            s_dim=ARGS._s_dim if ARGS.cond_decoder else 0,
        )
    LOGGER.info("Encoding shape: {}", enc_shape)

    recon_loss_fn_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.recon_loss == "l1":
        recon_loss_fn_ = nn.L1Loss(reduction="sum")
    elif ARGS.recon_loss == "l2":
        recon_loss_fn_ = nn.MSELoss(reduction="sum")
    elif ARGS.recon_loss == "huber":
        recon_loss_fn_ = lambda x, y: F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
    elif ARGS.recon_loss == "ce":
        recon_loss_fn_ = PixelCrossEntropy(reduction="sum")
    elif ARGS.recon_loss == "mixed":
        assert feature_group_slices is not None, "can only do multi loss with feature groups"
        recon_loss_fn_ = MixedLoss(feature_group_slices, reduction="sum")
    else:
        raise ValueError(f"{ARGS.recon_loss} is an invalid reconstruction loss")

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.vgg_weight != 0:
        vgg_loss = VGGLoss()
        vgg_loss.to(ARGS._device)

        def recon_loss_fn(input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return recon_loss_fn_(input_, target) + ARGS.vgg_weight * vgg_loss(input_, target)

    else:
        recon_loss_fn = recon_loss_fn_

    if not ARGS.stochastic:
        vae = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            decode_with_s=True,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        vae = VAE(
            encoder=encoder,
            decoder=decoder,
            kl_weight=ARGS.kl_weight,
            decode_with_s=True,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    vae.to(args._device)

    # Initialise Discriminator
    disc_fn = linear_disciminator

    disc_optimizer_kwargs = {"lr": args.disc_lr}

    disc_enc_y_kwargs = {
        "hidden_channels": ARGS.disc_enc_y_channels,
        "num_blocks": ARGS.disc_enc_y_depth,
        "use_bn": True,
    }

    disc_enc_y = build_discriminator(
        input_shape=enc_shape,
        target_dim=datasets.s_dim,
        train_on_recon=ARGS.train_on_recon,
        frac_enc=ARGS.enc_y_dim / enc_shape[0],
        model_fn=disc_fn,
        model_kwargs=disc_enc_y_kwargs,
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    disc_enc_y.to(args._device)

    disc_enc_s = None
    if ARGS.enc_s_dim > 0:

        disc_enc_s_kwargs = {
            "hidden_channels": ARGS.disc_enc_s_channels,
            "num_blocks": ARGS.disc_enc_s_depth,
            "use_bn": True,
        }
        disc_enc_s = build_discriminator(
            enc_shape,
            target_dim=datasets.s_dim,
            train_on_recon=ARGS.train_on_recon,
            frac_enc=ARGS.enc_s_dim / enc_shape[0],
            model_fn=disc_fn,
            model_kwargs=disc_enc_s_kwargs,
            optimizer_kwargs=disc_optimizer_kwargs,
        )

        disc_enc_s.to(args._device)

    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring model from checkpoint")
        vae, start_epoch = restore_model(ARGS, Path(ARGS.resume), vae)
        if ARGS.evaluate:
            log_metrics(ARGS, model=vae, data=datasets, save_to_csv=Path(ARGS.save_dir), step=0)
            return vae

    # Logging
    # wandb.set_model_graph(str(vae))
    LOGGER.info("Number of trainable parameters: {}", count_parameters(vae))

    best_loss = float("inf")
    n_vals_without_improvement = 0
    super_val_freq = ARGS.super_val_freq or ARGS.val_freq

    itr = 0
    # Train vae for N epochs
    for epoch in range(start_epoch, start_epoch + ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(vae, disc_enc_y, disc_enc_s, train_loader, epoch, recon_loss_fn)

        if epoch % ARGS.val_freq == 0:
            val_loss = validate(vae, disc_enc_y, disc_enc_s, val_loader, itr, recon_loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(args, save_dir, vae, epoch=epoch, sha=sha, best=True)
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

            LOGGER.info(
                "[VAL] Epoch {:04d} | Val Loss {:.6f} | "
                "No improvement during validation: {:02d}",
                epoch,
                val_loss,
                n_vals_without_improvement,
            )
        if ARGS.super_val and epoch % super_val_freq == 0:
            log_metrics(ARGS, model=vae, data=datasets, step=itr)
            save_model(args, save_dir, vae=vae, epoch=epoch, sha=sha)

    LOGGER.info("Training has finished.")
    path = save_model(args, save_dir, vae=vae, epoch=epoch, sha=sha)
    vae, _ = restore_model(args, path, vae=vae)
    log_metrics(ARGS, model=vae, data=datasets, save_to_csv=Path(ARGS.save_dir), step=itr)
    return vae


if __name__ == "__main__":
    main()
