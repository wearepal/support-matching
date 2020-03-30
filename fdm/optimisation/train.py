"""Main training file"""
import time
from itertools import islice
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast, Literal

import git
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import wandb

from fdm.configs import VaeArgs
from fdm.data import DatasetTriplet, load_dataset
from fdm.models import VAE, AutoEncoder, Classifier, EncodingSize, build_discriminator, Regressor
from fdm.models.configs import (
    conv_autoencoder,
    fc_autoencoder,
    fc_net,
    strided_28x28_net,
    residual_64x64_net,
)
from fdm.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    inf_generator,
    product,
    random_seed,
    readable_duration,
    wandb_log,
)

from .evaluation import log_metrics
from .loss import MixedLoss, PixelCrossEntropy, VGGLoss
from .utils import get_data_dim, log_images, restore_model, save_model

__all__ = ["main"]

ARGS: VaeArgs = None  # type: ignore[assignment]
LOGGER: Logger = None  # type: ignore[assignment]


def update_disc(
    x_c: Tensor,
    x_t: Tensor,
    generator: AutoEncoder,
    discriminator: Classifier,
    disc_distinguish: Optional[Regressor],
) -> Tuple[Tensor, float]:
    """Train the discriminator while keeping the generator constant.

    Args:
        x_c: x from the context set
        x_t: x from the training set
    """
    generator.eval()
    discriminator.train()
    if disc_distinguish is not None:
        disc_distinguish.train()

    ones = x_c.new_ones((x_c.size(0),))
    zeros = x_t.new_zeros((x_t.size(0),))
    invariances: List[Literal["s", "y"]] = ["s", "y"] if ARGS.three_way_split else ["s"]

    for _ in range(ARGS.num_disc_updates):
        encoding_t = generator.encode(x_t)
        encoding_c = generator.encode(x_c)
        disc_input_c: Tensor
        if ARGS.train_on_recon:
            disc_input_c = generator.decode(encoding_c).detach()  # just reconstruct

        disc_loss = x_c.new_zeros(())
        disc_loss_distinguish = x_c.new_zeros(())
        # in case of the three-way split, we have to check more than one invariance
        for invariance in invariances:
            disc_input_t = get_disc_input(generator, encoding_t, invariant_to=invariance)
            disc_input_t = disc_input_t.detach()
            if not ARGS.train_on_recon:
                disc_input_c = get_disc_input(generator, encoding_c, invariant_to=invariance)
                disc_input_c = disc_input_c.detach()

            # discriminator is trained to distinguish `disc_input_c` and `disc_input_t`
            disc_loss_true, disc_acc_true = discriminator.routine(disc_input_c, ones)
            disc_loss_false, disc_acc_false = discriminator.routine(disc_input_t, zeros)
            disc_loss += disc_loss_true + disc_loss_false
            if ARGS.three_way_split and not ARGS.train_on_recon:
                assert disc_distinguish is not None
                encs = generator.split_encoding(encoding_c)
                target_enc = encs.zs if invariance == "s" else encs.zy
                # predict target_enc from the encoding that should be invariant of it
                disc_loss_distinguish_c, _ = disc_distinguish.routine(disc_input_c, target_enc)
                disc_loss_distinguish += disc_loss_distinguish_c
        discriminator.zero_grad()
        disc_loss.backward()
        discriminator.step()

        if disc_distinguish is not None:
            disc_distinguish.zero_grad()
            disc_loss_distinguish.backward()
            disc_distinguish.step()
    return disc_loss, 0.5 * (disc_acc_true + disc_acc_false)  # statistics from last step


def update(
    x_c: Tensor,
    x_t: Tensor,
    generator: AutoEncoder,
    discriminator: Classifier,
    recon_loss_fn,
    disc_distinguish: Optional[Regressor],
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute all losses.

    Args:
        x_t: x from the training set
    """
    # Compute losses for the generator.
    discriminator.eval()
    if disc_distinguish is not None:
        disc_distinguish.eval()
    generator.train()
    logging_dict = {}

    # ================================ recon loss for training set ================================
    if ARGS.vae:
        generator = cast(VAE, generator)
        encoding, posterior = generator.encode(x_t, return_posterior=True)
        kl_div = generator.compute_divergence(encoding, posterior)
        kl_div /= x_t.size(0)
        kl_div *= ARGS.kl_weight
    else:
        encoding = generator.encode(x_t)
        kl_div = x_t.new_zeros(())

    recon_all = generator.decode(encoding)
    recon_loss = recon_loss_fn(recon_all, x_t)
    recon_loss /= x_t.size(0)
    elbo = recon_loss + kl_div

    # ================================ recon loss for context set =================================
    # we need a reconstruction loss for x_c because...
    # ...when we train on encodings, the network will otherwise just falsify encodings for x_c
    # ...when we train on recons, the GAN loss has it too easy to distinguish the two
    if ARGS.vae:
        generator = cast(VAE, generator)
        encoding_c, posterior_c = generator.encode(x_c, return_posterior=True)
        kl_div_c = generator.compute_divergence(encoding_c, posterior_c)
        kl_div_c /= x_c.size(0)
        kl_div_c *= ARGS.kl_weight
    else:
        encoding_c = generator.encode(x_c)
        kl_div_c = x_c.new_zeros(())

    recon_all_c = generator.decode(encoding_c)
    recon_loss_c = recon_loss_fn(recon_all_c, x_c)
    recon_loss_c /= x_c.size(0)
    recon_loss += recon_loss_c  # for logging
    kl_div += kl_div_c  # for logging
    elbo += recon_loss_c + kl_div_c
    elbo *= 0.5  # take average of the two recon losses

    # ==================================== adversarial losses =====================================
    disc_input = get_disc_input(generator, encoding, invariant_to="s")
    zeros = x_t.new_zeros((x_t.size(0),))
    disc_loss, disc_acc_inv_s = discriminator.routine(disc_input, zeros)

    disc_loss_distinguish = x_t.new_zeros(())
    if ARGS.three_way_split:
        disc_input_y = get_disc_input(generator, encoding, invariant_to="y")
        disc_loss_y, disc_acc_inv_y = discriminator.routine(disc_input_y, zeros)
        disc_loss += disc_loss_y
        logging_dict["Accuracy Disc 2"] = disc_acc_inv_y

        if not ARGS.train_on_recon:
            assert disc_distinguish is not None
            encs_c = generator.split_encoding(encoding_c)
            # predict zs from zn and zy
            disc_input_c = get_disc_input(generator, encoding_c, invariant_to="s")
            disc_loss_distinguish_s, _ = disc_distinguish.routine(disc_input_c, encs_c.zs)
            disc_loss_distinguish += disc_loss_distinguish_s
            logging_dict["Loss Distinguisher 1"] = disc_loss_distinguish_s.item()

            # predict zy from zn and zs
            disc_input_c_y = get_disc_input(generator, encoding_c, invariant_to="y")
            disc_loss_distinguish_y, _ = disc_distinguish.routine(disc_input_c_y, encs_c.zy)
            disc_loss_distinguish += disc_loss_distinguish_y
            logging_dict["Loss Distinguisher 2"] = disc_loss_distinguish_y.item()

    elbo *= ARGS.elbo_weight
    disc_loss *= ARGS.pred_s_weight
    disc_loss_distinguish *= ARGS.distinguish_weight

    gen_loss = elbo - disc_loss - disc_loss_distinguish

    # Update the generator's parameters
    generator.zero_grad()
    gen_loss.backward()
    generator.step()

    logging_dict.update(
        {
            "ELBO": elbo.item(),
            "Loss Adversarial": disc_loss.item(),
            "Accuracy Disc": disc_acc_inv_s,
            "KL divergence": kl_div.item(),
            "Loss Reconstruction": recon_loss.item(),
            "Loss Generator": gen_loss.item(),
        }
    )

    return gen_loss, logging_dict


def get_disc_input(
    generator: AutoEncoder, encoding: Tensor, invariant_to: Literal["s", "y"] = "s"
) -> Tensor:
    """Construct the input that the discriminator expects."""
    if ARGS.train_on_recon:
        zs_m, zy_m = generator.mask(encoding, random=True)
        recon = generator.decode(zs_m if invariant_to == "s" else zy_m)
        if ARGS.recon_loss == "ce":
            recon = recon.argmax(dim=1).float() / 255
            if ARGS.dataset != "cmnist":
                recon = recon * 2 - 1
        return recon
    else:
        zs_m, zy_m = generator.mask(encoding)
        return zs_m if invariant_to == "s" else zy_m


def train(
    generator: AutoEncoder,
    discriminator: Classifier,
    disc_distinguish: Optional[Regressor],
    context_data: DataLoader,
    train_data: DataLoader,
    epoch: int,
    recon_loss_fn,
) -> int:
    generator.train()

    total_loss_meter = AverageMeter()
    loss_meters: Optional[Dict[str, AverageMeter]] = None

    time_meter = AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    epoch_len = max(len(context_data), len(train_data))
    itr = start_itr = (epoch - 1) * epoch_len
    # FIXME: Should move from epoch- to iteration-based training.
    data_iterator = islice(zip(inf_generator(context_data), inf_generator(train_data)), epoch_len)

    for itr, ((x_c, _, _), (x_t, _, _)) in enumerate(data_iterator, start=start_itr):

        x_c, x_t = to_device(x_c, x_t)

        if itr >= ARGS.warmup_steps:
            # Train the discriminator on its own for a number of iterations
            update_disc(x_c, x_t, generator, discriminator, disc_distinguish)

        gen_loss, logging_dict = update(
            x_c=x_c,
            x_t=x_t,
            generator=generator,
            discriminator=discriminator,
            recon_loss_fn=recon_loss_fn,
            disc_distinguish=disc_distinguish,
        )

        # Log losses
        total_loss_meter.update(gen_loss.item())
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
                log_recons(generator=generator, x_t=x_t, itr=itr)

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


def to_device(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS._device, non_blocking=True) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)


def log_recons(generator: AutoEncoder, x_t: Tensor, itr: int, prefix: Optional[str] = None):
    """Log reconstructed images"""
    if ARGS.vae:
        generator = cast(VAE, generator)
        encoding = generator.encode(x_t[:64], stochastic=False)
    else:
        encoding = generator.encode(x_t[:64])

    recon = generator.all_recons(encoding, discretize=True)

    log_images(ARGS, x_t[:64], "original_x", step=itr, prefix=prefix)
    log_images(ARGS, recon.all, "reconstruction_all", step=itr, prefix=prefix)
    log_images(ARGS, recon.rand_s, "reconstruction_rand_s", step=itr, prefix=prefix)
    log_images(ARGS, recon.zero_s, "reconstruction_zero_s", step=itr, prefix=prefix)
    log_images(ARGS, recon.just_s, "reconstruction_just_s", step=itr, prefix=prefix)
    if ARGS.three_way_split:
        log_images(ARGS, recon.rand_y, "reconstruction_rand_2", step=itr, prefix=prefix)
        log_images(ARGS, recon.zero_y, "reconstruction_zero_2", step=itr, prefix=prefix)


def main(raw_args: Optional[List[str]] = None) -> AutoEncoder:
    """Main function

    Args:
        raw_args: commandline arguments
        datasets: a Dataset object

    Returns:
        the trained generator
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    args = VaeArgs(explicit_bool=True, underscores_to_dashes=True, fromfile_prefix_chars="@")
    args.parse_args(raw_args)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER
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
        "Size of context-set: {}, training-set: {}, test-set: {}",
        len(datasets.context),
        len(datasets.train),
        len(datasets.test),
    )
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    context_loader = DataLoader(
        datasets.context,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        datasets.train,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    # tesr_loader = DataLoader(
    #     datasets.test,
    #     shuffle=False,
    #     batch_size=ARGS.test_batch_size,
    #     num_workers=ARGS.num_workers,
    #     pin_memory=True,
    # )

    # ==== construct networks ====
    input_shape = get_data_dim(context_loader)
    is_image_data = len(input_shape) > 2

    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    if is_image_data:
        decoding_dim = input_shape[0] * 256 if args.recon_loss == "ce" else input_shape[0]
        # if ARGS.recon_loss == "ce":
        decoder_out_act = None
        # else:
        #     decoder_out_act = nn.Sigmoid() if ARGS.dataset == "cmnist" else nn.Tanh()
        encoder, decoder, enc_shape = conv_autoencoder(
            input_shape,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_channels,
            decoding_dim=decoding_dim,
            levels=ARGS.levels,
            decoder_out_act=decoder_out_act,
            variational=ARGS.vae,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            input_shape,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_channels,
            levels=ARGS.levels,
            variational=ARGS.vae,
        )
    zs_dim = round(ARGS.zs_frac * enc_shape[0])
    zy_dim = zs_dim if ARGS.three_way_split else 0
    zn_dim = enc_shape[0] - zs_dim - zy_dim
    encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim, zn=zn_dim)
    LOGGER.info("Encoding shape: {}, {}", enc_shape, encoding_size)

    recon_loss_fn_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.recon_loss == "l1":
        recon_loss_fn_ = nn.L1Loss(reduction="sum")
    elif ARGS.recon_loss == "l2":
        recon_loss_fn_ = nn.MSELoss(reduction="sum")
    elif ARGS.recon_loss == "bce":
        recon_loss_fn_ = nn.BCELoss(reduction="sum")
    elif ARGS.recon_loss == "huber":
        recon_loss_fn_ = lambda x, y: 0.1 * F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
    elif ARGS.recon_loss == "ce":
        recon_loss_fn_ = PixelCrossEntropy(reduction="sum")
    elif ARGS.recon_loss == "mixed":
        assert feature_group_slices is not None, "can only do multi gen_loss with feature groups"
        recon_loss_fn_ = MixedLoss(feature_group_slices, reduction="sum")
    else:
        raise ValueError(f"{ARGS.recon_loss} is an invalid reconstruction gen_loss")

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.vgg_weight != 0:
        vgg_loss = VGGLoss()
        vgg_loss.to(ARGS._device)

        def recon_loss_fn(input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return recon_loss_fn_(input_, target) + ARGS.vgg_weight * vgg_loss(input_, target)

    else:
        recon_loss_fn = recon_loss_fn_

    generator: AutoEncoder
    if ARGS.vae:
        generator = VAE(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            std_transform=ARGS.std_transform,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    else:
        generator = AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_args,
        )
    generator.to(args._device)

    # Initialise Discriminator
    disc_optimizer_kwargs = {"lr": args.disc_lr}
    disc_kwargs = {}
    disc_input_shape: Tuple[int, ...] = input_shape if ARGS.train_on_recon else enc_shape
    # FIXME: Architectures need to be GAN specific (e.g. incorporate spectral norm)
    if is_image_data and ARGS.train_on_recon:
        if args.dataset == "cmnist":
            disc_fn = strided_28x28_net
        else:
            disc_fn = residual_64x64_net
        disc_kwargs["batch_norm"] = False
    else:
        disc_fn = fc_net
        disc_kwargs["hidden_dims"] = args.disc_hidden_dims
        disc_input_shape = (product(disc_input_shape),)  # fc_net first flattens the input

    discriminator = build_discriminator(
        input_shape=disc_input_shape,
        target_dim=1,  # real vs fake
        model_fn=disc_fn,
        model_kwargs=disc_kwargs,
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    discriminator.to(args._device)

    disc_distinguish = None
    if args.three_way_split and not ARGS.train_on_recon:  # for now, only with train on encodings
        disc_dist_fn = fc_net
        disc_dist_kwargs = {"hidden_dims": args.disc_hidden_dims}
        output_dim = product((encoding_size.zy,) + enc_shape[1:])
        disc_distinguish = Regressor(
            disc_dist_fn(disc_input_shape, output_dim, **disc_dist_kwargs), disc_optimizer_kwargs,
        )
        disc_distinguish.to(args._device)

    def _spectral_norm(m):
        if hasattr(m, "weight"):
            return torch.nn.utils.spectral_norm(m)

    discriminator.apply(_spectral_norm)

    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        generator, start_epoch = restore_model(ARGS, Path(ARGS.resume), generator)
        if ARGS.evaluate:
            log_metrics(
                ARGS, model=generator, data=datasets, save_to_csv=Path(ARGS.save_dir), step=0
            )
            return generator

    # Logging
    # wandb.set_model_graph(str(generator))
    LOGGER.info("Number of trainable parameters: {}", count_parameters(generator))

    # best_loss = float("inf")
    n_vals_without_improvement = 0
    super_val_freq = ARGS.super_val_freq or ARGS.val_freq

    itr = 0
    # Train generator for N epochs
    for epoch in range(start_epoch, start_epoch + ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(
            generator=generator,
            discriminator=discriminator,
            disc_distinguish=disc_distinguish,
            context_data=context_loader,
            train_data=train_loader,
            epoch=epoch,
            recon_loss_fn=recon_loss_fn,
        )

        # if epoch % ARGS.val_freq == 0:
        #     val_loss = validate(generator, discriminator, train_loader, itr, recon_loss_fn)

        #     if val_loss < best_loss:
        #         best_loss = val_loss
        #         save_model(args, save_dir, generator, epoch=epoch, sha=sha, best=True)
        #         n_vals_without_improvement = 0
        #     else:
        #         n_vals_without_improvement += 1

        #     LOGGER.info(
        #         "[VAL] Epoch {:04d} | Val Loss {:.6f} | "
        #         "No improvement during validation: {:02d}",
        #         epoch,
        #         val_loss,
        #         n_vals_without_improvement,
        #     )
        if ARGS.super_val and epoch % super_val_freq == 0:
            log_metrics(ARGS, model=generator, data=datasets, step=itr)
            save_model(args, save_dir, model=generator, epoch=epoch, sha=sha)

    LOGGER.info("Training has finished.")
    path = save_model(args, save_dir, model=generator, epoch=epoch, sha=sha)
    generator, _ = restore_model(args, path, model=generator)
    log_metrics(ARGS, model=generator, data=datasets, save_to_csv=Path(ARGS.save_dir), step=itr)
    return generator


if __name__ == "__main__":
    main()
