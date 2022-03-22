"""Main training file"""
from __future__ import annotations
from collections import defaultdict
import logging
from pathlib import Path
import time
from typing import cast

from hydra.utils import to_absolute_path
import numpy as np
from ranzen.torch import random_seed
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models import resnet18, resnet50
import wandb
import yaml

from clustering.models import (
    BaseModel,
    CosineSimThreshold,
    Encoder,
    FactorizedModel,
    JointModel,
    Method,
    PseudoLabelEnc,
    PseudoLabelEncNoNorm,
    PseudoLabeler,
    PseudoLabelOutput,
    RankingStatistics,
    SelfSupervised,
    build_classifier,
)
from shared.configs import (
    ClusterConfig,
    ClusteringLabel,
    ClusteringMethod,
    Config,
    EncoderConfig,
    EncoderType,
    MiscConfig,
    PlMethod,
)
from shared.data.data_loading import DataModule, load_data
from shared.data.dataset_wrappers import RotationPrediction
from shared.data.misc import adaptive_collate
from shared.models.configs.classifiers import FcNet
from shared.utils import (
    AverageMeter,
    ClusterResults,
    as_pretty_dict,
    count_parameters,
    flatten_dict,
    get_class_id,
    get_data_dim,
    print_metrics,
    readable_duration,
    save_results,
)

from .build import build_ae
from .evaluation import classify_dataset
from .k_means import train as train_k_means
from .utils import cluster_metrics, count_occurances, get_cluster_label_path

__all__ = ["main"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Experiment(ExperimentBase):
    """Experiment singleton class."""

    def __init__(
        self,
        args: ClusterConfig,
        cfg: Config,
        data: DatasetConfig,
        enc: EncoderConfig,
        misc: MiscConfig,
        model: BaseModel,
        s_dim: int,
        y_dim: int,
    ) -> None:
        super().__init__(cfg=cfg, data_cfg=data, misc_cfg=misc)
        self.args = args
        self.model = model
        self.enc_conf = enc
        self.s_dim = s_dim
        self.y_dim = y_dim

    def train(self, context_data: DataLoader, train_data: DataLoader, epoch: int) -> int:
        total_loss_meter = AverageMeter()
        loss_meters = defaultdict(AverageMeter)

        time_meter = AverageMeter()
        start_epoch_time = time.time()
        end = start_epoch_time
        epoch_len = min(len(context_data), len(train_data))
        itr = start_itr = (epoch - 1) * epoch_len
        data_iterator = zip(context_data, train_data)
        self.model.train()

        for itr, ((x_c, _, _), (x_t, s_t, y_t)) in enumerate(data_iterator, start=start_itr):

            x_c, x_t, y_t, s_t = self.to_device(x_c, x_t, y_t, s_t)

            if self.args.with_supervision:
                loss_sup, logging_sup = self.model.supervised_loss(
                    x=x_t,
                    s=s_t,
                    y=y_t,
                    ce_weight=self.args.sup_ce_weight,
                    bce_weight=self.args.sup_bce_weight,
                )
            else:
                loss_sup = x_t.new_zeros(())
                logging_sup = {}
            loss_unsup, logging_unsup = self.model.unsupervised_loss(x_c)
            loss = loss_sup + loss_unsup

            self.model.zero_grad()
            loss.backward()
            self.model.step()

            # Log losses
            logging_dict = {**logging_unsup, **logging_sup}
            total_loss_meter.update(loss.item())
            for name, value in logging_dict.items():
                loss_meters[name].update(value)

            time_for_batch = time.time() - end
            time_meter.update(time_for_batch)

            wandb.log(logging_dict, step=itr)
            end = time.time()

        time_for_epoch = time.time() - start_epoch_time
        to_log = "[TRN] Epoch {:04d} | Duration: {} | Batches/s: {:.4g} | {} ({:.5g})".format(
            epoch,
            readable_duration(time_for_epoch),
            1 / time_meter.avg,
            " | ".join(f"{name}: {meter.avg:.5g}" for name, meter in loss_meters.items()),
            total_loss_meter.avg,
        )
        LOGGER.info(to_log)
        return itr

    def validate(
        self, val_data: DataLoader
    ) -> tuple[float, dict[str, float], dict[str, float | str]]:
        self.model.eval()
        to_cluster = self.args.cluster
        y_count = self.y_dim if self.y_dim > 1 else 2
        s_count = self.s_dim if self.s_dim > 1 else 2
        if to_cluster is ClusteringLabel.s:
            num_clusters = s_count
        elif to_cluster is ClusteringLabel.y:
            num_clusters = y_count
        elif to_cluster is ClusteringLabel.both:
            num_clusters = s_count * y_count
        else:
            num_clusters = cast(int, self.args.num_clusters)
        counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
        num_total = 0
        cluster_ids: list[np.ndarray] = []
        class_ids: list[Tensor] = []

        with torch.set_grad_enabled(False):
            for (x_v, s_v, y_v) in val_data:
                x_v = self.to_device(x_v)
                logits, _ = self.model(x_v)
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
            y_count=y_count,
            to_cluster=to_cluster,
        )

    def convert_and_save_results(
        self,
        cluster_label_path: Path,
        results: tuple[Tensor, Tensor, Tensor],
        enc_path: Path,
        context_metrics: dict[str, float] | None,
        test_metrics: dict[str, float] | None = None,
    ) -> Path:
        clusters, s, y = results
        s_count = self.s_dim if self.s_dim > 1 else 2
        class_ids = get_class_id(s=s, y=y, s_count=s_count, to_cluster=self.args.cluster)
        cluster_results = ClusterResults(
            flags=flatten_dict(as_pretty_dict(self.cfg)),
            cluster_ids=clusters,
            class_ids=class_ids,
            enc_path=enc_path,
            context_metrics=context_metrics,
            test_metrics=test_metrics,
        )
        return save_results(save_path=cluster_label_path, cluster_results=cluster_results)

    def save_model(self, save_dir: Path, epoch: int, sha: str, best: bool = False) -> Path:
        if best:
            filename = save_dir / "checkpt_best.pth"
        else:
            filename = save_dir / f"checkpt_epoch{epoch}.pth"
        save_dict = {
            "args": flatten_dict(as_pretty_dict(self.cfg)),
            "sha": sha,
            "model": self.model.state_dict(),
            "epoch": epoch,
        }

        torch.save(save_dict, filename)

        return filename

    def restore_model(self, filename: Path) -> tuple[BaseModel, int]:
        chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
        args_chkpt = chkpt["args"]
        assert self.enc_conf.levels == args_chkpt["enc.levels"]
        self.model.load_state_dict(chkpt["model"])
        return self.model, chkpt["epoch"]


def main(cfg: Config, cluster_label_file: Path | None = None) -> None:
    """Main function

    Args:
        hydra_config: configuration object from hydra
        cluster_label_file: path to a pth file with cluster IDs

    Returns:
        the trained generator
    """
    # ==== initialize config shorthands ====
    args = cfg.clust
    data = cfg.dm
    enc = cfg.enc
    misc = cfg.misc

    # ==== current git commit ====
    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha
    sha = ""  # this doesn't work with ray

    use_gpu = torch.cuda.is_available() and misc.gpu >= 0
    random_seed(misc.seed, use_cuda=use_gpu)
    if cluster_label_file is not None:
        misc.cluster_label_file = str(cluster_label_file)

    group = ""
    if misc.log_method:
        group += misc.log_method
    if misc.exp_group:
        group += "." + misc.exp_group
    if cfg.split.log_dataset:
        group += "." + cfg.split.log_dataset
    local_dir = Path(".", "local_logging")
    local_dir.mkdir(exist_ok=True)
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="fcm-hydra",
        dir=str(local_dir),
        config=flatten_dict(as_pretty_dict(cfg)),
        group=group if group else None,
        reinit=True,
        mode=misc.wandb.name,
    )
    run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

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
    datasets: DataModule = load_data(cfg)
    LOGGER.info(
        "Size of context-set: {}, training-set: {}, test-set: {}".format(
            len(datasets.context),
            len(datasets.train),
            len(datasets.test),
        )
    )
    args.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
    context_batch_size = round(args.batch_size * len(datasets.context) / len(datasets.train))
    context_loader = DataLoader(
        datasets.context,
        shuffle=True,
        batch_size=context_batch_size,
        num_workers=data.num_workers,
        pin_memory=True,
    )
    enc_train_data = ConcatDataset([datasets.context, datasets.train])
    if args.encoder == EncoderType.rotnet:
        enc_train_loader = DataLoader(
            RotationPrediction(enc_train_data, apply_all=True),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=data.num_workers,
            pin_memory=True,
            collate_fn=adaptive_collate,
        )
    else:
        enc_train_loader = DataLoader(
            enc_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=data.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        datasets.train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.test,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=data.num_workers,
        pin_memory=True,
    )

    # ==== construct networks ====
    input_shape = get_data_dim(context_loader)
    s_count = datasets.dim_s if datasets.dim_s > 1 else 2
    y_count = datasets.dim_y if datasets.dim_y > 1 else 2
    if args.cluster is ClusteringLabel.s:
        num_clusters = s_count
    elif args.cluster is ClusteringLabel.y:
        num_clusters = y_count
    elif args.cluster is ClusteringLabel.both:
        num_clusters = s_count * y_count
    elif args.cluster is ClusteringLabel.manual:
        assert args.num_clusters is not None
        if args.num_clusters < s_count * y_count:
            raise ValueError("It is not possible to 'undercluster' right now.")
        num_clusters = args.num_clusters
    else:
        raise ValueError("unknown clustering target")

    LOGGER.info(
        f"Number of clusters: {num_clusters}, accuracy computed with respect to {args.cluster.name}"
    )
    mappings: list[str] = []
    for i in range(num_clusters):
        if args.cluster is ClusteringLabel.s:
            mappings.append(f"{i}: s = {i}")
        elif args.cluster is ClusteringLabel.y:
            mappings.append(f"{i}: y = {i}")
        else:
            # class_id = y * s_count + s
            mappings.append(f"{i}: (y = {i // s_count}, s = {i % s_count})")
            if args.cluster is ClusteringLabel.manual and i + 1 == s_count * y_count:
                break
    LOGGER.info("class IDs:\n\t" + "\n\t".join(mappings))
    feature_group_slices = getattr(datasets.context, "feature_group_slices", None)

    # ================================= encoder =================================
    encoder: Encoder
    enc_dim: int
    if args.encoder in (EncoderType.ae, EncoderType.vae):
        encoder, enc_dim = build_ae(cfg, input_shape, feature_group_slices)
    else:
        if len(input_shape) < 2:
            raise ValueError("RotNet can only be applied to image data.")
        enc_optimizer_kwargs = {"lr": args.enc_lr, "weight_decay": args.enc_wd}
        enc_kwargs = {"pretrained": False, "num_classes": 4, "zero_init_residual": True}
        net = resnet18(**enc_kwargs) if isinstance(data, CmnistConfig) else resnet50(**enc_kwargs)

        encoder = SelfSupervised(model=net, num_classes=4, optimizer_kwargs=enc_optimizer_kwargs)
        enc_dim = 512
    encoder.to(device)

    LOGGER.info(f"Encoding shape: {enc_dim}")

    enc_path: Path
    if args.enc_path:
        enc_path = Path(args.enc_path)
        if args.encoder is EncoderType.rotnet:
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        save_dict = torch.load(args.enc_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(save_dict["encoder"])
        if "args" in save_dict:
            args_encoder = save_dict["args"]
            assert args.encoder.name == args_encoder["encoder_type"]
            assert enc.levels == args_encoder["levels"]
    else:
        encoder.fit(
            enc_train_loader, epochs=args.enc_epochs, device=device, use_wandb=args.enc_wandb
        )
        if args.encoder is EncoderType.rotnet:
            assert isinstance(encoder, SelfSupervised)
            encoder = encoder.get_encoder()
        # the args names follow the convention of the standalone VAE commandline args
        args_encoder = {"encoder_type": args.encoder.name, "levels": enc.levels}
        enc_path = save_dir.resolve() / "encoder"
        torch.save({"encoder": encoder.state_dict(), "args": args_encoder}, enc_path)
        LOGGER.info(f"To make use of this encoder:\n--enc-path {enc_path}")
        if args.enc_wandb:
            LOGGER.info("Stopping here because W&B will be messed up...")
            run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
            return

    cluster_label_path = get_cluster_label_path(misc, save_dir)
    if args.method == ClusteringMethod.kmeans:
        kmeans_results = train_k_means(
            cfg, encoder, datasets.context, num_clusters, s_count, y_count, enc_path=enc_path
        )
        # save_results(save_path=cluster_label_path, cluster_results=kmeans_results)
        run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
        return
    if args.finetune_encoder:
        encoder.freeze_initial_layers(
            args.freeze_layers, {"lr": args.finetune_lr, "weight_decay": args.weight_decay}
        )

    # ================================= labeler =================================
    pseudo_labeler: PseudoLabeler
    if args.pseudo_labeler == PlMethod.ranking:
        pseudo_labeler = RankingStatistics(k_num=args.k_num)
    elif args.pseudo_labeler == PlMethod.cosine:
        pseudo_labeler = CosineSimThreshold(
            upper_threshold=args.upper_threshold, lower_threshold=args.lower_threshold
        )
    else:
        raise ValueError("Unknown pseudo labeler")

    # ================================= method =================================
    method: Method
    if args.method == ClusteringMethod.pl_enc:
        method = PseudoLabelEnc()
    elif args.method == ClusteringMethod.pl_output:
        method = PseudoLabelOutput()
    elif args.method == ClusteringMethod.pl_enc_no_norm:
        method = PseudoLabelEncNoNorm()
    else:
        raise ValueError("Unknown method")

    # ================================= classifier =================================
    clf_optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    clf_fn = FcNet(hidden_dims=args.cl_hidden_dims)
    clf_input_shape = (enc_dim,)  # FcNet first flattens the input

    model: BaseModel
    if args.factorized_s_y:
        s_classifier = build_classifier(
            clf_input_shape, s_count, model_fn=clf_fn, optimizer_kwargs=clf_optimizer_kwargs
        )
        s_classifier.to(device)
        y_classifier = build_classifier(
            clf_input_shape, y_count, model_fn=clf_fn, optimizer_kwargs=clf_optimizer_kwargs
        )
        y_classifier.to(device)
        model = FactorizedModel(
            encoder=encoder,
            s_classifier=s_classifier,
            y_classifier=y_classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=args.finetune_encoder,
        )
    else:
        classifier = build_classifier(
            clf_input_shape, num_clusters, model_fn=clf_fn, optimizer_kwargs=clf_optimizer_kwargs
        )
        classifier.to(device)
        model = JointModel(
            encoder=encoder,
            classifier=classifier,
            method=method,
            pseudo_labeler=pseudo_labeler,
            train_encoder=args.finetune_encoder,
            to_cluster=args.cluster,
            s_count=s_count,
        )

    exp = Experiment(
        args=args,
        cfg=cfg,
        data=data,
        enc=enc,
        misc=misc,
        model=model,
        s_dim=datasets.dim_s,
        y_dim=datasets.dim_y,
    )

    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Resume from checkpoint
    if misc.resume is not None:
        LOGGER.info("Restoring generator from checkpoint")
        model, start_epoch = exp.restore_model(Path(misc.resume))
        if misc.evaluate:
            pth_path = exp.convert_and_save_results(
                cluster_label_path,
                classify_dataset(cfg, model, datasets.context),
                enc_path=enc_path,
                context_metrics={},  # TODO: compute this
            )
            run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
            return model, pth_path

    # Logging
    # wandb.set_model_graph(str(generator))
    num_parameters = count_parameters(model)
    LOGGER.info(f"Number of trainable parameters: {num_parameters}")

    # best_loss = float("inf")
    best_acc = 0.0
    n_vals_without_improvement = 0
    # super_val_freq = args.super_val_freq or args.val_freq

    itr = 0
    # Train generator for N epochs
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if n_vals_without_improvement > args.early_stopping > 0:
            break

        itr = exp.train(context_data=context_loader, train_data=train_loader, epoch=epoch)

        if epoch % args.val_freq == 0:
            val_acc, _, val_log = exp.validate(val_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                exp.save_model(save_dir, epoch=epoch, sha=sha, best=True)
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

            prepare = (
                f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in val_log.items()
            )
            LOGGER.info(
                "[VAL] Epoch {:04d} | {} | "
                "No improvement during validation: {:02d}".format(
                    epoch,
                    " | ".join(prepare),
                    n_vals_without_improvement,
                )
            )
            wandb.log(val_log, step=itr)
        # if args.super_val and epoch % super_val_freq == 0:
        #     log_metrics(args, model=model.bundle, data=datasets, step=itr)
        #     save_model(args, save_dir, model=model.bundle, epoch=epoch, sha=sha)

    LOGGER.info("Training has finished.")
    # path = save_model(args, save_dir, model=model, epoch=epoch, sha=sha)
    # model, _ = restore_model(args, path, model=model)
    _, test_metrics, _ = exp.validate(val_loader)
    _, context_metrics, _ = exp.validate(context_loader)
    LOGGER.info("Test metrics:")
    print_metrics({f"Test {k}": v for k, v in test_metrics.items()})
    LOGGER.info("Context metrics:")
    print_metrics({f"Context {k}": v for k, v in context_metrics.items()})
    pth_path = exp.convert_and_save_results(
        cluster_label_path=cluster_label_path,
        results=classify_dataset(cfg, model, datasets.context),
        enc_path=enc_path,
        context_metrics=context_metrics,
        test_metrics=test_metrics,
    )
    run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process
