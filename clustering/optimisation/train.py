"""Main training file"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import git
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models import resnet18, resnet50

from clustering.models import (
    Classifier,
    CosineSimThreshold,
    Encoder,
    Method,
    Model,
    MultiHeadModel,
    PseudoLabelEnc,
    PseudoLabelEncNoNorm,
    PseudoLabeler,
    PseudoLabelOutput,
    RankingStatistics,
    SelfSupervised,
    build_classifier,
)
from shared.configs import (
    CL,
    DS,
    PL,
    ClusterArgs,
    Config,
    DatasetConfig,
    Enc,
    EncoderConfig,
    Meth,
    Misc,
)
from shared.data.data_loading import DatasetTriplet, load_dataset
from shared.data.dataset_wrappers import RotationPrediction
from shared.data.misc import adaptive_collate
from shared.models.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import (
    AverageMeter,
    ModelFn,
    count_parameters,
    flatten,
    get_data_dim,
    print_metrics,
    prod,
    random_seed,
    readable_duration,
    save_results,
    wandb_log,
)

from .build import build_ae
from .evaluation import classify_dataset
from .k_means import train as train_k_means
from .utils import (
    cluster_metrics,
    convert_and_save_results,
    count_occurances,
    get_class_id,
    get_cluster_label_path,
    restore_model,
    save_model,
)

__all__ = ["main"]

ARGS: ClusterArgs = None  # type: ignore[assignment]
CFG: Config = None  # type: ignore[assignment]
DATA: DatasetConfig = None  # type: ignore[assignment]
ENC: EncoderConfig = None  # type: ignore[assignment]
MISC: Misc = None  # type: ignore[assignment]


def main(
    cfg: Config, cluster_label_file: Optional[Path] = None, use_wandb: Optional[bool] = None
) -> Tuple[Model, Path]:
    """Main function

    Args:
        cluster_label_file: path to a pth file with cluster IDs
        use_wandb: this arguments overwrites the flag

    Returns:
        the trained generator
    """
    # ==== initialize globals ====
    global ARGS, CFG, DATA, ENC, MISC
    ARGS = cfg.clust
    CFG = cfg
    DATA = cfg.data
    ENC = cfg.enc
    MISC = cfg.misc

    # ==== current git commit ====
    if os.environ.get("STARTED_BY_GUILDAI", None) == "1":
        sha = ""
    else:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

    use_gpu = torch.cuda.is_available() and MISC.gpu >= 0
    random_seed(MISC.seed, use_gpu)
    if cluster_label_file is not None:
        MISC.cluster_label_file = str(cluster_label_file)

    if use_wandb is not None:
        MISC.use_wandb = use_wandb
    if MISC.use_wandb:
        group = ""
        if MISC.log_method:
            group += MISC.log_method
        if MISC.exp_group:
            group += "." + MISC.exp_group
        if cfg.bias.log_dataset:
            group += "." + cfg.bias.log_dataset
        wandb.init(
            entity="predictive-analytics-lab",
            project="fcm-hydra",
            config=flatten(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)),
            group=group if group else None,
        )

    save_dir = Path(to_absolute_path(MISC.save_dir)) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    print(str(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)))
    print(f"Save directory: {save_dir.resolve()}")
    # ==== check GPU ====
    MISC._device = f"cuda:{MISC.gpu}" if use_gpu else "cpu"
    device = torch.device(MISC._device)
    print(f"{torch.cuda.device_count()} GPUs available. Using device '{device}'")

    # ==== construct dataset ====
    datasets: DatasetTriplet = load_dataset(CFG)
    print(
        "Size of context-set: {}, training-set: {}, test-set: {}".format(
            len(datasets.context),
            len(datasets.train),
            len(datasets.test),
        )
    )
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    context_batch_size = round(ARGS.batch_size * len(datasets.context) / len(datasets.train))
    context_loader = DataLoader(
        datasets.context,
        shuffle=True,
        batch_size=context_batch_size,
        num_workers=MISC.num_workers,
        pin_memory=True,
    )
    enc_train_data = ConcatDataset([datasets.context, datasets.train])
    if ARGS.encoder == Enc.rotnet:
        enc_train_loader = DataLoader(
            RotationPrediction(enc_train_data, apply_all=True),
            shuffle=True,
            batch_size=ARGS.batch_size,
            num_workers=MISC.num_workers,
            pin_memory=True,
            collate_fn=adaptive_collate,
        )
    else:
        enc_train_loader = DataLoader(
            enc_train_data,
            shuffle=True,
            batch_size=ARGS.batch_size,
            num_workers=MISC.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        datasets.train,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=MISC.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.test,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=MISC.num_workers,
        pin_memory=True,
    )

    # ==== construct networks ====
    input_shape = get_data_dim(context_loader)
    s_count = datasets.s_dim if datasets.s_dim > 1 else 2
    y_count = datasets.y_dim if datasets.y_dim > 1 else 2
    if ARGS.cluster == CL.s:
        num_clusters = s_count
    elif ARGS.cluster == CL.y:
        num_clusters = y_count
    else:
        num_clusters = s_count * y_count
    print(
        f"Number of clusters: {num_clusters}, accuracy computed with respect to {ARGS.cluster.name}"
    )
    mappings: List[str] = []
    for i in range(num_clusters):
        if ARGS.cluster == CL.s:
            mappings.append(f"{i}: s = {i}")
        elif ARGS.cluster == CL.y:
            mappings.append(f"{i}: y = {i}")
        else:
            # class_id = y * s_count + s
            mappings.append(f"{i}: (y = {i // s_count}, s = {i % s_count})")
    print("class IDs:\n\t" + "\n\t".join(mappings))
    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    # ================================= encoder =================================
    encoder: Encoder
    enc_shape: Tuple[int, ...]
    if ARGS.encoder in (Enc.ae, Enc.vae):
        encoder, enc_shape = build_ae(CFG, input_shape, feature_group_slices)
    else:
        if len(input_shape) < 2:
            raise ValueError("RotNet can only be applied to image data.")
        enc_optimizer_kwargs = {"lr": ARGS.enc_lr, "weight_decay": ARGS.enc_wd}
        enc_kwargs = {"pretrained": False, "num_classes": 4, "zero_init_residual": True}
        net = resnet18(**enc_kwargs) if DATA.dataset == DS.cmnist else resnet50(**enc_kwargs)

        encoder = SelfSupervised(model=net, num_classes=4, optimizer_kwargs=enc_optimizer_kwargs)
        enc_shape = (512,)
        encoder.to(device)

    print(f"Encoding shape: {enc_shape}")

    enc_path: Path
    if ARGS.enc_path:
        enc_path = Path(ARGS.enc_path)
        if ARGS.encoder == Enc.rotnet:
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        save_dict = torch.load(ARGS.enc_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert ARGS.encoder.name == args_encoder["encoder_type"]
            assert ENC.levels == args_encoder["levels"]
    else:
        encoder.fit(
            enc_train_loader, epochs=ARGS.enc_epochs, device=device, use_wandb=ARGS.enc_wandb
        )
        if ARGS.encoder == Enc.rotnet:
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        # the args names follow the convention of the standalone VAE commandline args
        args_encoder = {"encoder_type": ARGS.encoder.name, "levels": ENC.levels}
        enc_path = save_dir.resolve() / "encoder"
        torch.save({"encoder": encoder.state_dict(), "args": args_encoder}, enc_path)
        print(f"To make use of this encoder:\n--enc-path {enc_path}")
        if ARGS.enc_wandb:
            print("Stopping here because W&B will be messed up...")
            return

    cluster_label_path = get_cluster_label_path(MISC, save_dir)
    if ARGS.method == Meth.kmeans:
        kmeans_results = train_k_means(
            CFG, encoder, datasets.context, num_clusters, s_count, enc_path
        )
        pth = save_results(save_path=cluster_label_path, cluster_results=kmeans_results)
        return (), pth
    if ARGS.finetune_encoder:
        encoder.freeze_initial_layers(
            ARGS.freeze_layers, {"lr": ARGS.finetune_lr, "weight_decay": ARGS.weight_decay}
        )

    # ================================= labeler =================================
    pseudo_labeler: PseudoLabeler
    if ARGS.pseudo_labeler == PL.ranking:
        pseudo_labeler = RankingStatistics(k_num=ARGS.k_num)
    elif ARGS.pseudo_labeler == PL.cosine:
        pseudo_labeler = CosineSimThreshold(
            upper_threshold=ARGS.upper_threshold, lower_threshold=ARGS.lower_threshold
        )

    # ================================= method =================================
    method: Method
    if ARGS.method == Meth.pl_enc:
        method = PseudoLabelEnc()
    elif ARGS.method == Meth.pl_output:
        method = PseudoLabelOutput()
    elif ARGS.method == Meth.pl_enc_no_norm:
        method = PseudoLabelEncNoNorm()

    # ================================= classifier =================================
    clf_optimizer_kwargs = {"lr": ARGS.lr, "weight_decay": ARGS.weight_decay}
    clf_fn = FcNet(hidden_dims=ARGS.cl_hidden_dims)
    clf_input_shape = (prod(enc_shape),)  # FcNet first flattens the input

    classifier = build_classifier(
        input_shape=clf_input_shape,
        target_dim=s_count if ARGS.use_multi_head else num_clusters,
        model_fn=clf_fn,
        optimizer_kwargs=clf_optimizer_kwargs,
        num_heads=y_count if ARGS.use_multi_head else 1,
    )
    classifier.to(device)

    model: Union[Model, MultiHeadModel]
    if ARGS.use_multi_head:
        labeler_fn: ModelFn
        if DATA.dataset == DS.cmnist:
            labeler_fn = Mp32x23Net(batch_norm=True)
        elif DATA.dataset == DS.celeba:
            labeler_fn = Mp64x64Net(batch_norm=True)
        else:
            labeler_fn = FcNet(hidden_dims=ARGS.labeler_hidden_dims)

        labeler_optimizer_kwargs = {"lr": ARGS.labeler_lr, "weight_decay": ARGS.labeler_wd}
        labeler: Classifier = build_classifier(
            input_shape=input_shape,
            target_dim=s_count,
            model_fn=labeler_fn,
            optimizer_kwargs=labeler_optimizer_kwargs,
        )
        labeler.to(device)
        print("Fitting the labeler to the labeled data.")
        labeler.fit(
            train_loader,
            epochs=ARGS.labeler_epochs,
            device=device,
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
    if MISC.resume is not None:
        print("Restoring generator from checkpoint")
        model, start_epoch = restore_model(CFG, Path(MISC.resume), model)
        if MISC.evaluate:
            pth_path = convert_and_save_results(
                CFG,
                cluster_label_path,
                classify_dataset(CFG, model, datasets.context),
                enc_path=enc_path,
                context_metrics={},  # TODO: compute this
            )
            return model, pth_path

    # Logging
    # wandb.set_model_graph(str(generator))
    num_parameters = count_parameters(model)
    print(f"Number of trainable parameters: {num_parameters}")

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
            val_acc, _, val_log = validate(model, val_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                save_model(CFG, save_dir, model, epoch=epoch, sha=sha, best=True)
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

            prepare = (
                f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in val_log.items()
            )
            print(
                "[VAL] Epoch {:04d} | {} | "
                "No improvement during validation: {:02d}".format(
                    epoch,
                    " | ".join(prepare),
                    n_vals_without_improvement,
                )
            )
            wandb_log(MISC, val_log, step=itr)
        # if ARGS.super_val and epoch % super_val_freq == 0:
        #     log_metrics(ARGS, model=model.bundle, data=datasets, step=itr)
        #     save_model(args, save_dir, model=model.bundle, epoch=epoch, sha=sha)

    print("Training has finished.")
    # path = save_model(args, save_dir, model=model, epoch=epoch, sha=sha)
    # model, _ = restore_model(args, path, model=model)
    _, test_metrics, _ = validate(model, val_loader)
    _, context_metrics, _ = validate(model, context_loader)
    print("Test metrics:")
    print_metrics({f"Test {k}": v for k, v in test_metrics.items()})
    print("Context metrics:")
    print_metrics({f"Context {k}": v for k, v in context_metrics.items()})
    pth_path = convert_and_save_results(
        CFG,
        cluster_label_path=cluster_label_path,
        results=classify_dataset(CFG, model, datasets.context),
        enc_path=enc_path,
        context_metrics=context_metrics,
        test_metrics=test_metrics,
    )
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
    s_count = MISC._s_dim if MISC._s_dim > 1 else 2

    for itr, ((x_c, _, _), (x_t, s_t, y_t)) in enumerate(data_iterator, start=start_itr):

        x_c, x_t, y_t, s_t = to_device(x_c, x_t, y_t, s_t)

        if ARGS.with_supervision and not ARGS.use_multi_head:
            class_id = get_class_id(s=s_t, y=y_t, s_count=s_count, to_cluster=ARGS.cluster)
            loss_sup, logging_sup = model.supervised_loss(
                x_t, class_id, ce_weight=ARGS.sup_ce_weight, bce_weight=ARGS.sup_bce_weight
            )
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

        wandb_log(MISC, logging_dict, step=itr)
        end = time.time()

    time_for_epoch = time.time() - start_epoch_time
    assert loss_meters is not None
    to_log = "[TRN] Epoch {:04d} | Duration: {} | Batches/s: {:.4g} | {} ({:.5g})".format(
        epoch,
        readable_duration(time_for_epoch),
        1 / time_meter.avg,
        " | ".join(f"{name}: {meter.avg:.5g}" for name, meter in loss_meters.items()),
        total_loss_meter.avg,
    )
    print(to_log)
    return itr


def validate(
    model: Model, val_data: DataLoader
) -> Tuple[float, Dict[str, float], Dict[str, Union[float, str]]]:
    model.eval()
    to_cluster = ARGS.cluster
    y_count = MISC._y_dim if MISC._y_dim > 1 else 2
    s_count = MISC._s_dim if MISC._s_dim > 1 else 2
    if to_cluster == CL.s:
        num_clusters = s_count
    elif to_cluster == CL.y:
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

    cluster_ids_np = np.concatenate(cluster_ids, axis=0)
    true_class_ids = torch.cat(class_ids).numpy()
    return cluster_metrics(
        cluster_ids=cluster_ids_np,
        counts=counts,
        true_class_ids=true_class_ids,
        num_total=num_total,
        s_count=s_count,
        to_cluster=to_cluster,
    )


def to_device(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(torch.device(MISC._device), non_blocking=True) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)
