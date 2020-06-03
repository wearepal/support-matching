"""Main training file"""
from __future__ import annotations
import time
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import git
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, resnet50
import wandb

from shared.data.dataset_wrappers import RotationPrediction
from shared.data.data_loading import load_dataset, DatasetTriplet
from shared.data.misc import adaptive_collate
from shared.models.configs.classifiers import fc_net, mp_64x64_net, mp_32x32_net
from shared.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    prod,
    random_seed,
    readable_duration,
    save_results,
    wandb_log,
    get_data_dim,
)
from clustering.configs import ClusterArgs
from clustering.models import (
    Classifier,
    CosineSimThreshold,
    Encoder,
    PseudoLabeler,
    Method,
    Model,
    MultiHeadModel,
    PseudoLabelEnc,
    PseudoLabelEncNoNorm,
    PseudoLabelOutput,
    RankingStatistics,
    SelfSupervised,
    build_classifier,
)

from .evaluation import classify_dataset
from .utils import restore_model, save_model, find_assignment, count_occurances, get_class_id
from .build import build_ae
from .k_means import train as train_k_means

__all__ = ["main"]

ARGS: ClusterArgs = None  # type: ignore[assignment]
LOGGER: Logger = None  # type: ignore[assignment]


def main(raw_args: Optional[List[str]] = None, known_only: bool = True) -> Tuple[Model, Path]:
    """Main function

    Args:
        raw_args: commandline arguments

    Returns:
        the trained generator
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    args = ClusterArgs(fromfile_prefix_chars="@").parse_args(raw_args, known_only=known_only)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER
    ARGS = args

    if ARGS.use_wandb:
        wandb.init(entity="predictive-analytics-lab", project="fcm", config=args.as_dict())

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
    enc_train_data = ConcatDataset([datasets.context, datasets.train])
    if args.encoder == "rotnet":
        enc_train_loader = DataLoader(
            RotationPrediction(enc_train_data, apply_all=True),
            shuffle=True,
            batch_size=ARGS.batch_size,
            num_workers=ARGS.num_workers,
            pin_memory=True,
            collate_fn=adaptive_collate,
        )
    else:
        enc_train_loader = DataLoader(
            enc_train_data,
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
    mappings: List[str] = []
    for i in range(num_clusters):
        if ARGS.cluster == "s":
            mappings.append(f"{i}: s = {i}")
        elif ARGS.cluster == "y":
            mappings.append(f"{i}: y = {i}")
        else:
            # class_id = y * s_count + s
            mappings.append(f"{i}: (y = {i // s_count}, s = {i % s_count})")
    LOGGER.info("class IDs:\n\t" + "\n\t".join(mappings))
    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    # ================================= encoder =================================
    encoder: Encoder
    enc_shape: Tuple[int, ...]
    if ARGS.encoder in ("ae", "vae"):
        encoder, enc_shape = build_ae(ARGS, input_shape, feature_group_slices)
    else:
        if len(input_shape) < 2:
            raise ValueError("RotNet can only be applied to image data.")
        enc_optimizer_kwargs = {"lr": args.enc_lr, "weight_decay": args.enc_wd}
        enc_kwargs = {"pretrained": False, "num_classes": 4, "zero_init_residual": True}
        net = resnet18(**enc_kwargs) if args.dataset == "cmnist" else resnet50(**enc_kwargs)

        encoder = SelfSupervised(model=net, num_classes=4, optimizer_kwargs=enc_optimizer_kwargs)
        enc_shape = (512,)
        encoder.to(args._device)

    LOGGER.info("Encoding shape: {}", enc_shape)

    if args.enc_path:
        if args.encoder == "rotnet":
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        save_dict = torch.load(args.enc_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert args.encoder == args_encoder["encoder_type"]
            assert args.enc_levels == args_encoder["levels"]
    else:
        encoder.fit(
            enc_train_loader, epochs=args.enc_epochs, device=args._device, use_wandb=ARGS.enc_wandb
        )
        if args.encoder == "rotnet":
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        # the args names follow the convention of the standalone VAE commandline args
        args_encoder = {"encoder_type": args.encoder, "levels": args.enc_levels}
        torch.save({"encoder": encoder.state_dict(), "args": args_encoder}, save_dir / "encoder")
        LOGGER.info("To make use of this encoder:\n--enc-path {}", save_dir.resolve() / "encoder")
        if ARGS.enc_wandb:
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
    if ARGS.pseudo_labeler == "ranking":
        pseudo_labeler = RankingStatistics(k_num=ARGS.k_num)
    elif ARGS.pseudo_labeler == "cosine":
        pseudo_labeler = CosineSimThreshold(
            upper_threshold=ARGS.upper_threshold, lower_threshold=ARGS.lower_threshold
        )

    # ================================= method =================================
    method: Method
    if ARGS.method == "pl_enc":
        method = PseudoLabelEnc()
    elif ARGS.method == "pl_output":
        method = PseudoLabelOutput()
    elif ARGS.method == "pl_enc_no_norm":
        method = PseudoLabelEncNoNorm()

    # ================================= classifier =================================
    clf_optimizer_kwargs = {"lr": ARGS.lr, "weight_decay": ARGS.weight_decay}
    clf_kwargs = {}
    clf_fn = fc_net
    clf_kwargs["hidden_dims"] = args.cl_hidden_dims
    clf_input_shape = (prod(enc_shape),)  # fc_net first flattens the input

    classifier = build_classifier(
        input_shape=clf_input_shape,
        target_dim=s_count if ARGS.use_multi_head else num_clusters,
        model_fn=clf_fn,
        model_kwargs=clf_kwargs,
        optimizer_kwargs=clf_optimizer_kwargs,
        num_heads=y_count if ARGS.use_multi_head else 1,
    )
    classifier.to(args._device)

    model: Union[Model, MultiHeadModel]
    if ARGS.use_multi_head:
        labeler_kwargs = {}
        if args.dataset == "cmnist":
            labeler_fn = mp_32x32_net
        elif args.dataset == "celeba":
            labeler_fn = mp_64x64_net
        else:
            labeler_fn = fc_net
            labeler_kwargs["hidden_dims"] = args.labeler_hidden_dims

        labeler_optimizer_kwargs = {"lr": ARGS.labeler_lr, "weight_decay": ARGS.labeler_wd}
        clf_fn = fc_net
        clf_kwargs["hidden_dims"] = args.cl_hidden_dims
        labeler: Classifier = build_classifier(
            input_shape=input_shape,
            target_dim=s_count,
            model_fn=labeler_fn,
            model_kwargs=labeler_kwargs,
            optimizer_kwargs=labeler_optimizer_kwargs,
        )
        labeler.to(args._device)
        LOGGER.info("Fitting the labeler to the labeled data.")
        labeler.fit(
            train_loader,
            epochs=ARGS.labeler_epochs,
            device=ARGS._device,
            use_wandb=ARGS.labeler_wandb,
        )
        labeler.eval()
        model = MultiHeadModel(
            encoder=encoder,
            classifiers=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            labeler=labeler,
            train_encoder=ARGS.finetune_encoder,
        )
    else:
        model = Model(
            encoder=encoder,
            classifier=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=ARGS.finetune_encoder,
        )

    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        model, start_epoch = restore_model(ARGS, Path(ARGS.resume), model)
        if ARGS.evaluate:
            pth_path = save_results(ARGS, classify_dataset(ARGS, model, datasets.context), save_dir)
            return model, pth_path

    # Logging
    # wandb.set_model_graph(str(generator))
    num_parameters = count_parameters(model)
    LOGGER.info("Number of trainable parameters: {}", num_parameters)

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
                save_model(args, save_dir, model, epoch=epoch, sha=sha, best=True)
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
    path = save_model(args, save_dir, model=model, epoch=epoch, sha=sha)
    model, _ = restore_model(args, path, model=model)
    validate(model, val_loader)
    pth_path = save_results(ARGS, classify_dataset(ARGS, model, datasets.context), save_dir)
    return model, pth_path


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
    s_count = ARGS._s_dim if ARGS._s_dim > 1 else 2

    for itr, ((x_c, s_c, y_c), (x_t, s_t, y_t)) in enumerate(data_iterator, start=start_itr):

        x_c, s_c, y_c, x_t, y_t, s_t = to_device(x_c, s_c, y_c, x_t, y_t, s_t)

        if ARGS.with_supervision and not ARGS.use_multi_head:
            class_id = get_class_id(s=s_t, y=y_t, s_count=s_count, to_cluster=ARGS.cluster)
            loss_sup, logging_sup = model.supervised_loss(
                x_t, class_id, ce_weight=ARGS.sup_ce_weight, bce_weight=ARGS.sup_bce_weight
            )
        else:
            loss_sup = x_t.new_zeros(())
            logging_sup = {}
            
        class_id = get_class_id(s=s_c, y=y_c, s_count=s_count, to_cluster=ARGS.cluster)
        loss_unsup, logging_unsup = model.supervised_loss(x_c, class_id, ce_weight=0.0, bce_weight=1)
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


def validate(model: Model, val_data: DataLoader) -> Tuple[float, Dict[str, Union[float, str]]]:
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
    cluster_ids: List[np.ndarray] = []
    class_ids: List[Tensor] = []

    with torch.set_grad_enabled(False):
        for (x_v, s_v, y_v) in val_data:
            x_v = to_device(x_v)
            logits = model(x_v)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            counts, class_id = count_occurances(counts, preds, s_v, y_v, s_count, to_cluster)
            num_total += y_v.size(0)
            cluster_ids.append(preds)
            class_ids.append(class_id)

    # find best assignment for cluster to classes
    best_acc, best_ass, logging_dict = find_assignment(counts, num_total)
    cluster_ids_np = np.concatenate(cluster_ids, axis=0)
    pred_class_ids = best_ass[cluster_ids_np]  # use the best assignment to get the class IDs
    true_class_ids = torch.cat(class_ids).numpy()
    conf_mat = confusion_matrix(true_class_ids, pred_class_ids, normalize="all")
    logging_dict["confusion matrix"] = f"\n{conf_mat}\n"
    return best_acc, logging_dict


def to_device(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS._device, non_blocking=True) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)


if __name__ == "__main__":
    main()
