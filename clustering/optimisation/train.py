"""Main training file"""
from __future__ import annotations
import time
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import git
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import wandb

from shared.data import DatasetTriplet, load_dataset
from shared.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    prod,
    random_seed,
    readable_duration,
    wandb_log,
)
from clustering.configs import ClusterArgs
from clustering.models import (
    Bundle,
    Classifier,
    CosineSimThreshold,
    Encoder,
    Labeler,
    Method,
    Model,
    PseudoLabelEnc,
    PseudoLabelEncNoNorm,
    PseudoLabelOutput,
    RankingStatistics,
    build_classifier,
)
from clustering.models.configs import fc_net

from .evaluation import log_metrics
from .utils import get_data_dim, restore_model, save_model, find_assignment, count_occurances
from .build import build_ae
from .k_means import train as train_k_means

__all__ = ["main"]

ARGS: ClusterArgs = None  # type: ignore[assignment]
LOGGER: Logger = None  # type: ignore[assignment]


def main(raw_args: Optional[List[str]] = None) -> Model:
    """Main function

    Args:
        raw_args: commandline arguments

    Returns:
        the trained generator
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    args = ClusterArgs(fromfile_prefix_chars="@")
    args.parse_args(raw_args)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER
    ARGS = args

    if ARGS.use_wandb:
        wandb.init(project="fcm", config=args.as_dict())

    save_dir = Path(ARGS.save_dir) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info(str(args))
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
    context_batch_size = round(ARGS.batch_size * len(datasets.context) / len(datasets.train))
    context_loader = DataLoader(
        datasets.context,
        shuffle=True,
        batch_size=context_batch_size,
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
    val_loader = DataLoader(
        datasets.test,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

    # ==== construct networks ====
    input_shape = get_data_dim(context_loader)
    s_count = datasets.s_dim if datasets.s_dim > 1 else 2
    y_count = datasets.y_dim if datasets.y_dim > 1 else 2
    if ARGS.cluster == "s":
        num_clusters = s_count
    elif ARGS.cluster == "y":
        num_clusters = y_count
    else:
        num_clusters = s_count * y_count
    LOGGER.info(
        "Number of clusters: {}, accuracy computed with respect to {}", num_clusters, ARGS.cluster
    )
    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    # ================================= encoder =================================
    encoder: Encoder
    enc_shape: Tuple[int, ...]
    if ARGS.encoder in ("ae", "vae"):
        encoder, enc_shape = build_ae(ARGS, input_shape, feature_group_slices)
    LOGGER.info("Encoding shape: {}", enc_shape)

    if args.enc_path:
        save_dict = torch.load(args.enc_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert args.encoder == args_encoder["encoder_type"]
            assert args.enc_levels == args_encoder["levels"]
    else:
        encoder.fit(
            context_loader, epochs=args.enc_epochs, device=args._device, use_wandb=ARGS.use_wandb
        )
        # the args names follow the convention of the standalone VAE commandline args
        args_encoder = {"encoder_type": args.encoder, "levels": args.enc_levels}
        torch.save({"encoder": encoder.state_dict(), "args": args_encoder}, save_dir / "encoder")
        if ARGS.use_wandb:
            LOGGER.info("Stopping here because W&B will be messed up...")
            return

    if ARGS.method == "kmeans":
        train_k_means(ARGS, encoder, datasets.context, num_clusters, s_count)
        return
    if ARGS.finetune_encoder:
        encoder.freeze_initial_layers(
            ARGS.freeze_layers, {"lr": ARGS.finetune_lr, "weight_decay": ARGS.weight_decay}
        )

    # ================================= labeler =================================
    labeler: Labeler
    if ARGS.labeler == "ranking":
        labeler = RankingStatistics(k_num=ARGS.k_num)
    elif ARGS.labeler == "cosine":
        labeler = CosineSimThreshold(
            upper_threshold=ARGS.upper_threshold, lower_threshold=ARGS.lower_threshold
        )

    # ================================= classifier =================================
    classifier: Classifier
    disc_optimizer_kwargs = {"lr": ARGS.lr, "weight_decay": ARGS.weight_decay}
    disc_kwargs = {}
    disc_fn = fc_net
    disc_kwargs["hidden_dims"] = args.cl_hidden_dims
    disc_input_shape = (prod(enc_shape),)  # fc_net first flattens the input
    classifier = build_classifier(
        input_shape=disc_input_shape,
        target_dim=num_clusters,
        model_fn=disc_fn,
        model_kwargs=disc_kwargs,
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    classifier.to(args._device)

    # ================================= method =================================
    method: Method
    if ARGS.method == "pl_enc":
        method = PseudoLabelEnc()
    elif ARGS.method == "pl_output":
        method = PseudoLabelOutput()
    elif ARGS.method == "pl_enc_no_norm":
        method = PseudoLabelEncNoNorm()

    model = Model(
        bundle=Bundle(encoder=encoder, labeler=labeler, classifier=classifier),
        method=method,
        train_encoder=ARGS.finetune_encoder,
    )

    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        bundle, start_epoch = restore_model(ARGS, Path(ARGS.resume), model.bundle)
        model = Model(bundle=bundle, method=method, train_encoder=ARGS.finetune_encoder)
        if ARGS.evaluate:
            log_metrics(ARGS, model=model, data=datasets, save_to_csv=Path(ARGS.save_dir), step=0)
            return model

    # Logging
    # wandb.set_model_graph(str(generator))
    LOGGER.info("Number of trainable parameters: {}", count_parameters(model.bundle))

    # best_loss = float("inf")
    best_acc = 0.0
    n_vals_without_improvement = 0
    # super_val_freq = ARGS.super_val_freq or ARGS.val_freq

    itr = 0
    # Train generator for N epochs
    for epoch in range(start_epoch, start_epoch + ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(model=model, context_data=context_loader, train_data=train_loader, epoch=epoch)

        if epoch % ARGS.val_freq == 0:
            val_acc, val_log = validate(model, val_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                save_model(args, save_dir, model.bundle, epoch=epoch, sha=sha, best=True)
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

            prepare = (
                f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in val_log.items()
            )
            LOGGER.info(
                "[VAL] Epoch {:04d} | {} | " "No improvement during validation: {:02d}",
                epoch,
                " | ".join(prepare),
                n_vals_without_improvement,
            )
            wandb_log(ARGS, val_log, step=itr)
        # if ARGS.super_val and epoch % super_val_freq == 0:
        #     log_metrics(ARGS, model=model.bundle, data=datasets, step=itr)
        #     save_model(args, save_dir, model=model.bundle, epoch=epoch, sha=sha)

    LOGGER.info("Training has finished.")
    path = save_model(args, save_dir, model=model.bundle, epoch=epoch, sha=sha)
    bundle, _ = restore_model(args, path, model=model.bundle)
    model = Model(bundle=bundle, method=method, train_encoder=ARGS.finetune_encoder)
    validate(model, val_loader, results_dir=save_dir)
    # log_metrics(ARGS, model=model, data=datasets, save_to_csv=Path(ARGS.save_dir), step=itr)
    return model


def train(model: Model, context_data: DataLoader, train_data: DataLoader, epoch: int) -> int:
    total_loss_meter = AverageMeter()
    loss_meters: Optional[Dict[str, AverageMeter]] = None

    time_meter = AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    epoch_len = min(len(context_data), len(train_data))
    itr = start_itr = (epoch - 1) * epoch_len
    data_iterator = zip(context_data, train_data)
    model.train()

    for itr, ((x_c, _, _), (x_t, _, y_t)) in enumerate(data_iterator, start=start_itr):

        x_c, x_t, y_t = to_device(x_c, x_t, y_t)

        if ARGS.with_supervision:
            loss_sup, logging_sup = model.supervised_loss(x_t, y_t)
        else:
            loss_sup = x_t.new_zeros(())
            logging_sup = {}
        loss_unsup, logging_unsup = model.unsupervised_loss(x_c)
        loss = loss_sup + loss_unsup

        model.zero_grad()
        loss.backward()
        model.step()

        # Log losses
        logging_dict = {**logging_unsup, **logging_sup}
        total_loss_meter.update(loss.item())
        if loss_meters is None:
            loss_meters = {name: AverageMeter() for name in logging_dict}
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        time_for_batch = time.time() - end
        time_meter.update(time_for_batch)

        wandb_log(ARGS, logging_dict, step=itr)
        end = time.time()

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


def validate(
    model: Model, val_data: DataLoader, results_dir: Optional[Path] = None
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    to_cluster = ARGS.cluster
    y_count = ARGS._y_dim if ARGS._y_dim > 1 else 2
    s_count = ARGS._s_dim if ARGS._s_dim > 1 else 2
    if to_cluster == "s":
        num_clusters = s_count
    elif to_cluster == "y":
        num_clusters = y_count
    else:
        num_clusters = s_count * y_count
    counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    num_total = 0
    if results_dir is not None:
        cluster_ids: List[np.ndarray] = []

    with torch.set_grad_enabled(False):
        for (x_v, s_v, y_v) in val_data:
            x_v = to_device(x_v)
            logits = model(x_v)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            counts = count_occurances(counts, preds, s_v, y_v, s_count, to_cluster)
            num_total += y_v.size(0)
            if results_dir is not None:
                cluster_ids.append(preds)

    # find best assignment for cluster to classes
    best_acc, best_ass, logging_dict = find_assignment(counts, num_total)

    if results_dir is not None:
        cluster_ids_np = np.concatenate(cluster_ids, axis=0)
        class_ids = best_ass[cluster_ids_np]  # use the best assignment to get the class IDs
        to_save: Dict[str, np.ndarray] = {}
        if to_cluster == "s":
            to_save["s"] = class_ids
        elif to_cluster == "y":
            to_save["y"] = class_ids
        else:
            # class_id = y * s_count + s
            to_save["s"] = class_ids % s_count
            to_save["y"] = class_ids // s_count
        save_path = results_dir / "cluster_results.npz"
        np.savez_compressed(save_path, **to_save)
        LOGGER.info("Saved results in {}", save_path)
    return best_acc, logging_dict


def to_device(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS._device, non_blocking=True) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)


if __name__ == "__main__":
    main()
