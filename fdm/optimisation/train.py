"""Main training file"""
import time
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from typing_extensions import Literal

from fdm.configs import FdmArgs
from fdm.models import AutoEncoder, Classifier, EncodingSize, PartitionedAeInn, build_discriminator
from fdm.models.configs import Residual64x64Net, Strided28x28Net
from fdm.optimisation.mmd import mmd2
from shared.data import DatasetTriplet, load_dataset
from shared.layers import Aggregator, AttentionAggregator, SimpleAggregator, SimpleAggregatorT
from shared.models.configs import conv_autoencoder, fc_autoencoder
from shared.models.configs.classifiers import FcNet, ModelAggregatorWrapper
from shared.utils import (
    AverageMeter,
    ModelFn,
    accept_prefixes,
    confirm_empty,
    count_parameters,
    get_logger,
    inf_generator,
    label_to_class_id,
    load_results,
    prod,
    random_seed,
    readable_duration,
    wandb_log,
)

from .build import build_ae, build_inn
from .evaluation import baseline_metrics, log_metrics
from .inn_training import InnComponents, update_disc_on_inn, update_inn
from .loss import MixedLoss, PixelCrossEntropy, VGGLoss
from .utils import (
    get_all_num_samples,
    log_images,
    restore_model,
    save_model,
    weight_for_balance,
    weights_with_counts,
)

__all__ = ["main"]

ARGS: FdmArgs = None  # type: ignore[assignment]
LOGGER: Logger = None  # type: ignore[assignment]
Generator = Union[AutoEncoder, PartitionedAeInn]


def main(cluster_label_file: Optional[Path] = None, initialize_wandb: bool = True) -> Generator:
    """Main function.

    Args:
        cluster_label_file: path to a pth file with cluster IDs
        initialize_wandb: if False, we assume that W&B has already been initialized

    Returns:
        the trained generator
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # args
    args = FdmArgs(fromfile_prefix_chars="@", explicit_bool=True, underscores_to_dashes=True)
    args.parse_args(accept_prefixes(("--a-", "--d-", "--e-")), known_only=True)
    confirm_empty(args.extra_args, to_ignore=("--b-", "--c-"))

    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    if cluster_label_file is not None:
        args.cluster_label_file = str(cluster_label_file)
    # ==== initialize globals ====
    global ARGS, LOGGER
    ARGS = args

    if ARGS.use_wandb:
        if initialize_wandb:
            wandb.init(entity="predictive-analytics-lab", project="fdm", config=args.as_dict())
        else:
            wandb.config.update(args.as_dict())

    save_dir = Path(ARGS.save_dir) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info(str(ARGS))
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
    dataloader_args: Dict[str, Any]
    s_count = max(datasets.s_dim, 2)

    cluster_results = None
    if ARGS.cluster_label_file:
        cluster_results = load_results(ARGS)
        ARGS._cluster_test_acc = cluster_results.test_acc
        ARGS._cluster_context_acc = cluster_results.context_acc
        weights, n_clusters, min_count, max_count = weight_for_balance(
            cluster_results.cluster_ids, min_size=None if ARGS.oversample else ARGS.batch_size
        )
        # if ARGS.oversample, oversample the smaller clusters instead of undersample the larger ones
        num_samples = n_clusters * max_count if ARGS.oversample else n_clusters * min_count
        assert num_samples > ARGS.batch_size, "not enough samples for a batch"
        context_sampler = WeightedRandomSampler(weights, num_samples, replacement=ARGS.oversample)
        dataloader_kwargs = dict(sampler=context_sampler)
    elif ARGS.balanced_context:
        context_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.context,
            s_count=s_count,
            test_batch_size=ARGS.test_batch_size,
            batch_size=ARGS.batch_size,
            num_workers=ARGS.num_workers,
            oversample=ARGS.oversample,
            balance_hierarchical=False,
        )
        dataloader_kwargs = dict(sampler=context_sampler, shuffle=False)
    else:
        dataloader_kwargs = dict(shuffle=True)

    context_loader = DataLoader(
        datasets.context,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
        drop_last=True,
        **dataloader_kwargs,
    )

    train_sampler = build_weighted_sampler_from_dataset(
        dataset=datasets.train,
        s_count=s_count,
        test_batch_size=ARGS.test_batch_size,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        oversample=ARGS.oversample,
        balance_hierarchical=True,
    )
    train_loader = DataLoader(
        dataset=datasets.train,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.test,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    context_data_itr = inf_generator(context_loader)
    train_data_itr = inf_generator(train_loader)
    # ==== construct networks ====
    input_shape = next(context_data_itr)[0][0].shape
    is_image_data = len(input_shape) > 2

    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    if is_image_data:
        decoding_dim = input_shape[0] * 256 if args.recon_loss == "ce" else input_shape[0]
        # if ARGS.recon_loss == "ce":
        decoder_out_act = None
        # else:
        #     decoder_out_act = nn.Sigmoid() if ARGS.dataset == "cmnist" else nn.Tanh()
        encoder, decoder, enc_shape = conv_autoencoder(
            input_shape,
            ARGS.enc_init_chans,
            encoding_dim=ARGS.enc_out_dim,
            decoding_dim=decoding_dim,
            levels=ARGS.enc_levels,
            decoder_out_act=decoder_out_act,
            variational=ARGS.vae,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            input_shape,
            ARGS.enc_init_chans,
            encoding_dim=ARGS.enc_out_dim,
            levels=ARGS.enc_levels,
            variational=ARGS.vae,
        )

    if ARGS.enc_snorm:

        def _snorm(_module: nn.Module) -> nn.Module:
            if hasattr(_module, "weight"):
                return torch.nn.utils.spectral_norm(_module)
            return _module

        encoder.apply(_snorm)

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

    generator: Generator
    if ARGS.use_inn:
        autoencoder = build_ae(
            ARGS, encoder, decoder, encoding_size=None, feature_group_slices=feature_group_slices
        )
        if prod(enc_shape) == enc_shape[0]:
            is_enc_image_data = False
            print("Encoding will not be treated as image data.")
        else:
            is_enc_image_data = is_image_data
        generator = build_inn(
            args=ARGS,
            autoencoder=autoencoder,
            ae_loss_fn=recon_loss_fn,
            is_image_data=is_enc_image_data,
            save_dir=save_dir,
            ae_enc_shape=enc_shape,
            context_loader=context_loader,
        )
        encoding_size = generator.encoding_size
    else:
        zs_dim = round(ARGS.zs_frac * enc_shape[0])
        zy_dim = enc_shape[0] - zs_dim
        encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim)
        generator = build_ae(
            args=ARGS,
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
        )
        # load pretrained encoder if one is provided
        if args.use_pretrained_enc and cluster_results is not None:
            save_dict = torch.load(
                cluster_results.enc_path, map_location=lambda storage, loc: storage
            )
            generator.load_state_dict(save_dict["encoder"])
            if "args" in save_dict:
                args_encoder = save_dict["args"]
                assert args_encoder["encoder_type"] == "vae" if args.vae else "ae"
                assert args_encoder["levels"] == args.enc_levels

    LOGGER.info("Encoding shape: {}, {}", enc_shape, encoding_size)

    # ================================== Initialise Discriminator =================================

    disc_optimizer_kwargs = {"lr": args.disc_lr}
    disc_input_shape: Tuple[int, ...] = input_shape if ARGS.train_on_recon else enc_shape
    # FIXME: Architectures need to be GAN specific (e.g. incorporate spectral norm)
    disc_fn: ModelFn
    if is_image_data and ARGS.train_on_recon:
        if args.dataset == "cmnist":
            disc_fn = Strided28x28Net(batch_norm=False)
        else:
            disc_fn = Residual64x64Net(batch_norm=False)
    else:
        disc_fn = FcNet(hidden_dims=ARGS.disc_hidden_dims)
        # FcNet first flattens the input
        disc_input_shape = (
            (prod(disc_input_shape),)
            if isinstance(disc_input_shape, Sequence)
            else disc_input_shape
        )

    if ARGS.batch_wise_loss != "none":
        final_proj = FcNet(ARGS.batch_wise_hidden_dims) if ARGS.batch_wise_hidden_dims else None
        aggregator: Aggregator
        if args.batch_wise_loss == "attention":
            aggregator = AttentionAggregator(args.batch_wise_latent, final_proj=final_proj)
        elif args.batch_wise_loss == "simple":
            aggregator = SimpleAggregator(latent_dim=args.batch_wise_latent, final_proj=final_proj)
        elif args.batch_wise_loss == "transposed":
            aggregator = SimpleAggregatorT(batch_dim=args.batch_size, final_proj=final_proj)

        disc_fn = ModelAggregatorWrapper(disc_fn, aggregator, embed_dim=args.batch_wise_latent)

    components: Union[AeComponents, InnComponents]
    disc: Classifier
    if not ARGS.use_inn:
        disc_list = []
        for k in range(ARGS.num_discs):
            disc = build_discriminator(
                input_shape=disc_input_shape,
                target_dim=1,  # real vs fake
                model_fn=disc_fn,
                optimizer_kwargs=disc_optimizer_kwargs,
            )
            disc_list.append(disc)
        disc_ensemble = nn.ModuleList(disc_list)
        disc_ensemble.to(args._device)

        predictor_y = build_discriminator(
            input_shape=(prod(enc_shape),),  # this is always trained on encodings
            target_dim=datasets.y_dim,
            model_fn=FcNet(hidden_dims=None),  # no hidden layers
            optimizer_kwargs=disc_optimizer_kwargs,
        )
        predictor_y.to(args._device)

        predictor_s = build_discriminator(
            input_shape=(prod(enc_shape),),  # this is always trained on encodings
            target_dim=datasets.s_dim,
            model_fn=FcNet(hidden_dims=None),  # no hidden layers
            optimizer_kwargs=disc_optimizer_kwargs,
        )
        predictor_s.to(args._device)

        components = AeComponents(
            generator=generator,
            disc_ensemble=disc_ensemble,
            recon_loss_fn=recon_loss_fn,
            predictor_y=predictor_y,
            predictor_s=predictor_s,
        )
    else:
        disc_list = []
        for k in range(ARGS.num_discs):
            disc = build_discriminator(
                input_shape=disc_input_shape,
                target_dim=1,  # real vs fake
                model_fn=disc_fn,
                optimizer_kwargs=disc_optimizer_kwargs,
            )
            disc_list.append(disc)
        disc_ensemble = nn.ModuleList(disc_list)
        disc_ensemble.to(args._device)

        # classifier for y
        class_fn: ModelFn
        if is_image_data:
            if args.dataset == "cmnist":
                class_fn = Strided28x28Net(batch_norm=False)
            else:
                class_fn = Residual64x64Net(batch_norm=False)
        else:
            class_fn = FcNet(hidden_dims=ARGS.disc_hidden_dims)
        predictor = None
        if ARGS.train_on_recon and ARGS.pred_y_weight > 0:
            predictor = build_discriminator(
                input_shape=input_shape,
                target_dim=args._y_dim,  # real vs fake
                model_fn=class_fn,
                optimizer_kwargs=disc_optimizer_kwargs,
            )
            predictor.to(args._device)
            predictor.fit(Subset(datasets.context, np.arange(100)), 50, ARGS._device, test_loader)
        components = InnComponents(inn=generator, disc_ensemble=disc_ensemble, predictor=predictor)

    start_itr = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        generator, start_itr = restore_model(ARGS, Path(ARGS.resume), generator)
        if ARGS.evaluate:
            log_metrics(ARGS, generator, datasets, 0, save_to_csv=Path(ARGS.save_dir))
            return generator

    # Logging
    LOGGER.info("Number of trainable parameters: {}", count_parameters(generator))

    itr = start_itr
    disc: nn.Module
    loss_meters: Optional[Dict[str, AverageMeter]] = None
    start_time = time.monotonic()

    for itr in range(start_itr, ARGS.iters + 1):

        logging_dict = train_step(
            components=components,
            context_data_itr=context_data_itr,
            train_data_itr=train_data_itr,
            itr=itr,
        )
        if loss_meters is None:
            loss_meters = {name: AverageMeter() for name in logging_dict}
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        if itr % ARGS.log_freq == 0:
            assert loss_meters is not None
            log_string = " | ".join(f"{name}: {loss.avg:.5g}" for name, loss in loss_meters.items())
            elapsed = time.monotonic() - start_time
            LOGGER.info(
                "[TRN] Iteration {:04d} | Elapsed: {} | Iterations/s: {:.4g} | {}",
                itr,
                readable_duration(elapsed),
                ARGS.log_freq / elapsed,
                log_string,
            )

            loss_meters = None
            start_time = time.monotonic()

        if ARGS.validate and itr % ARGS.val_freq == 0:
            if itr == ARGS.val_freq:  # first validation
                baseline_metrics(ARGS, datasets, save_to_csv=Path(ARGS.save_dir))
            log_metrics(ARGS, model=generator, data=datasets, step=itr)
            save_model(ARGS, save_dir, model=generator, itr=itr, sha=sha)

        if ARGS.disc_reset_prob > 0:
            for k, discriminator in enumerate(components.disc_ensemble):
                if np.random.uniform() < ARGS.disc_reset_prob:
                    LOGGER.info("Reinitializing discriminator {}", k)
                    discriminator.reset_parameters()

    LOGGER.info("Training has finished.")
    # path = save_model(args, save_dir, model=generator, epoch=epoch, sha=sha)
    # generator, _ = restore_model(args, path, model=generator)
    log_metrics(ARGS, model=generator, data=datasets, save_to_csv=Path(ARGS.save_dir), step=itr)
    return generator


def build_weighted_sampler_from_dataset(
    dataset: Dataset,
    s_count: int,
    oversample: bool,
    test_batch_size: int,
    batch_size: int,
    num_workers: int,
    balance_hierarchical: bool,
) -> WeightedRandomSampler:
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
        # here we make sure that in a batch, y is balanced and within the y subsets, s is balanced
        y_weights, y_unique_weights_counts = weights_with_counts(y_all.view(-1))
        quad_weights, quad_unique_weights_counts = weights_with_counts(class_ids)
        weights = y_weights * quad_weights

        all_num_samples = get_all_num_samples(
            quad_unique_weights_counts, y_unique_weights_counts, s_count
        )
        num_samples = max(all_num_samples) if oversample else min(all_num_samples)
    else:
        weights, n_clusters, min_count, max_count = weight_for_balance(class_ids)
        num_samples = n_clusters * max_count if oversample else n_clusters * min_count
    assert num_samples > batch_size, f"not enough training samples ({num_samples}) to fill a batch"
    return WeightedRandomSampler(weights.squeeze(), num_samples, replacement=oversample)


def get_batch(
    context_data_itr: Iterator[Tuple[Tensor, Tensor, Tensor]],
    train_data_itr: Iterator[Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x_c = to_device(next(context_data_itr)[0])
    x_t, s_t, y_t = to_device(*next(train_data_itr))
    return x_c, x_t, s_t, y_t


def train_step(
    components: Union["AeComponents", InnComponents],
    context_data_itr: Iterator[Tuple[Tensor, Tensor, Tensor]],
    train_data_itr: Iterator[Tuple[Tensor, Tensor, Tensor]],
    itr: int,
) -> Dict[str, float]:

    disc_weight = 0.0 if itr < ARGS.warmup_steps else ARGS.disc_weight
    if ARGS.disc_method == "nn":
        # Train the discriminator on its own for a number of iterations
        for _ in range(ARGS.num_disc_updates):
            x_c, x_t, s_t, y_t = get_batch(
                context_data_itr=context_data_itr, train_data_itr=train_data_itr
            )
            if components.type_ == "ae":
                update_disc(x_c, x_t, components, itr < ARGS.warmup_steps)
            else:
                update_disc_on_inn(ARGS, x_c, x_t, components, itr < ARGS.warmup_steps)

    x_c, x_t, s_t, y_t = get_batch(context_data_itr=context_data_itr, train_data_itr=train_data_itr)
    if components.type_ == "ae":
        _, logging_dict = update(
            x_c=x_c, x_t=x_t, s_t=s_t, y_t=y_t, ae=components, warmup=itr < ARGS.warmup_steps
        )
    else:
        _, logging_dict = update_inn(
            args=ARGS, x_c=x_c, x_t=x_t, models=components, disc_weight=disc_weight
        )

    wandb_log(ARGS, logging_dict, step=itr)

    # Log images
    if itr % ARGS.log_freq == 0:
        with torch.no_grad():
            if components.type_ == "ae":
                generator = components.generator
                x_log = x_t
            else:
                generator = components.inn
                x_log = x_c
            log_recons(generator=generator, x=x_log, itr=itr)
    return logging_dict


class AeComponents(NamedTuple):
    """Things needed to run the INN model."""

    generator: AutoEncoder
    disc_ensemble: nn.ModuleList
    recon_loss_fn: Callable
    predictor_y: Classifier
    predictor_s: Classifier
    type_: Literal["ae"] = "ae"


def update_disc(x_c: Tensor, x_t: Tensor, ae: AeComponents, warmup: bool = False) -> Tensor:
    """Train the discriminator while keeping the generator constant.

    Args:
        x_c: x from the context set
        x_t: x from the training set
    """
    ae.generator.eval()
    ae.predictor_y.eval()
    ae.predictor_s.eval()
    ae.disc_ensemble.train()

    if ARGS.batch_wise_loss == "none":
        ones = x_c.new_ones((x_c.size(0),))
        zeros = x_t.new_zeros((x_t.size(0),))
    else:
        ones = x_c.new_ones((1,))
        zeros = x_t.new_zeros((1,))
    invariances = ["s"]

    if not ARGS.vae:
        encoding_t = ae.generator.encode(x_t)
        if not ARGS.train_on_recon:
            encoding_c = ae.generator.encode(x_c)
    if ARGS.vae:
        encoding_t = ae.generator.encode(x_t, stochastic=True)
        if not ARGS.train_on_recon:
            encoding_c = ae.generator.encode(x_c, stochastic=True)

    if ARGS.train_on_recon:
        disc_input_c = x_c

    disc_loss = x_c.new_zeros(())
    for invariance in invariances:
        disc_input_t = get_disc_input(ae.generator, encoding_t, invariant_to=invariance)
        disc_input_t = disc_input_t.detach()
        if not ARGS.train_on_recon:
            disc_input_c = get_disc_input(ae.generator, encoding_c, invariant_to=invariance)
            disc_input_c = disc_input_c.detach()

        for discriminator in ae.disc_ensemble:
            disc_loss_true = discriminator.routine(disc_input_c, ones)[0]
            disc_loss_false = discriminator.routine(disc_input_t, zeros)[0]
            disc_loss += disc_loss_true + disc_loss_false
        disc_loss /= len(ae.disc_ensemble)
    if not warmup:
        for discriminator in ae.disc_ensemble:
            discriminator.zero_grad()
        disc_loss.backward()
        for discriminator in ae.disc_ensemble:
            discriminator.step()

    return disc_loss


def update(
    x_c: Tensor, x_t: Tensor, s_t: Tensor, y_t: Tensor, ae: AeComponents, warmup: bool
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute all losses.

    Args:
        x_t: x from the training set
    """
    disc_weight = 0.0 if warmup else ARGS.disc_weight
    # Compute losses for the generator.
    ae.predictor_y.train()
    ae.predictor_s.train()
    ae.generator.train()
    logging_dict = {}

    # ================================ recon loss for training set ================================
    encoding, elbo, logging_dict_elbo = ae.generator.routine(x_t, ae.recon_loss_fn, ARGS.kl_weight)

    # ================================ recon loss for context set =================================
    # we need a reconstruction loss for x_c because...
    # ...when we train on encodings, the network will otherwise just falsify encodings for x_c
    # ...when we train on recons, the GAN loss has it too easy to distinguish the two
    encoding_c, elbo_c, logging_dict_elbo_c = ae.generator.routine(
        x_c, ae.recon_loss_fn, ARGS.kl_weight
    )
    logging_dict.update({k: v + logging_dict_elbo_c[k] for k, v in logging_dict_elbo.items()})
    elbo = 0.5 * (elbo + elbo_c)  # take average of the two recon losses

    # ==================================== adversarial losses =====================================
    disc_input_no_s = get_disc_input(ae.generator, encoding, invariant_to="s")

    if ARGS.disc_method == "nn":
        if ARGS.batch_wise_loss == "none":
            zeros = x_t.new_zeros((x_t.size(0),))
        else:
            zeros = x_t.new_zeros((1,))

        disc_loss = x_t.new_zeros(())
        for discriminator in ae.disc_ensemble:
            discriminator.eval()
            disc_loss -= discriminator.routine(disc_input_no_s, zeros)[0]
        disc_loss /= len(ae.disc_ensemble)

    else:
        x = disc_input_no_s
        y = get_disc_input(ae.generator, encoding_c.detach(), invariant_to="s")
        disc_loss = mmd2(
            x=x,
            y=y,
            kernel=ARGS.mmd_kernel,
            scales=ARGS.mmd_scales,
            wts=ARGS.mmd_wts,
            add_dot=ARGS.mmd_add_dot,
        )
    pred_y_loss = x_t.new_zeros(())
    pred_s_loss = x_t.new_zeros(())
    # this is a pretty cheap masking operation, so it's okay if it's not used
    enc_no_s, enc_no_y = ae.generator.mask(encoding, random=False)
    if ARGS.pred_y_weight > 0:
        # predictor is on encodings; predict y from the part that is invariant to s
        pred_y_loss, pred_y_acc = ae.predictor_y.routine(enc_no_s, y_t)
        pred_y_loss *= ARGS.pred_y_weight
        logging_dict["Loss Predictor y"] = pred_y_loss.item()
        logging_dict["Accuracy Predictor y"] = pred_y_acc
    if ARGS.pred_s_weight > 0:
        pred_s_loss, pred_s_acc = ae.predictor_s.routine(enc_no_y, s_t)
        pred_s_loss *= ARGS.pred_s_weight
        logging_dict["Loss Predictor s"] = pred_s_loss.item()
        logging_dict["Accuracs Predictor s"] = pred_s_acc

    elbo *= ARGS.elbo_weight
    disc_loss *= disc_weight

    gen_loss = elbo + disc_loss + pred_y_loss + pred_s_loss
    # Update the generator's parameters
    ae.generator.zero_grad()
    if ARGS.pred_y_weight > 0:
        ae.predictor_y.zero_grad()
    if ARGS.pred_s_weight > 0:
        ae.predictor_s.zero_grad()
    gen_loss.backward()
    ae.generator.step()
    if ARGS.pred_y_weight > 0:
        ae.predictor_y.step()
    if ARGS.pred_s_weight > 0:
        ae.predictor_s.step()

    final_logging = {
        "ELBO": elbo.item(),
        "Loss Adversarial": disc_loss.item(),
        "Loss Generator": gen_loss.item(),
    }
    logging_dict.update(final_logging)

    return gen_loss, logging_dict


def get_disc_input(
    generator: AutoEncoder, encoding: Tensor, invariant_to: Literal["s", "y"] = "s"
) -> Tensor:
    """Construct the input that the discriminator expects."""
    if ARGS.train_on_recon:
        zs_m, zy_m = generator.mask(encoding, random=True)
        recon = generator.decode(zs_m if invariant_to == "s" else zy_m, mode="relaxed")
        if ARGS.recon_loss == "ce":
            recon = recon.argmax(dim=1).float() / 255
            if ARGS.dataset != "cmnist":
                recon = recon * 2 - 1
        return recon
    else:
        zs_m, zy_m = generator.mask(encoding)
        return zs_m if invariant_to == "s" else zy_m


def to_device(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS._device, non_blocking=True) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)


def log_recons(
    generator: Union[AutoEncoder, PartitionedAeInn],
    x: Tensor,
    itr: int,
    prefix: Optional[str] = None,
) -> None:
    """Log reconstructed images"""
    encoding = generator.encode(x[:64], stochastic=False)
    recon = generator.all_recons(encoding, mode="hard")

    log_images(ARGS, x[:64], "original_x", step=itr, prefix=prefix)
    log_images(ARGS, recon.all, "reconstruction_all", step=itr, prefix=prefix)
    log_images(ARGS, recon.rand_s, "reconstruction_rand_s", step=itr, prefix=prefix)
    log_images(ARGS, recon.zero_s, "reconstruction_zero_s", step=itr, prefix=prefix)
    log_images(ARGS, recon.just_s, "reconstruction_just_s", step=itr, prefix=prefix)


if __name__ == "__main__":
    main()
