from __future__ import annotations
import logging
from typing import Dict, NamedTuple, Optional, Sequence, Tuple
from typing_extensions import Literal

import ethicml as em
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
from tqdm import tqdm

from shared.configs import (
    AdultConfig,
    CelebaConfig,
    CmnistConfig,
    Config,
    EvalTrainData,
    ImageDatasetConfig,
    IsicConfig,
    WandbMode,
)
from shared.data import DatasetTriplet, TensorDataTupleDataset, adult, get_data_tuples
from shared.models.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import ModelFn, compute_metrics, make_tuple_from_data, prod
from suds.models import AutoEncoder, Classifier, SplitEncoding

from .utils import ExtractableDataset, build_weighted_sampler_from_dataset, log_images

__all__ = ["baseline_metrics", "log_metrics"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class InvarianceDatasets(NamedTuple):
    inv_y: TensorDataTupleDataset | None
    inv_s: TensorDataTupleDataset | None


def log_sample_images(cfg: Config, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(cfg, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    cfg: Config,
    model,
    data: DatasetTriplet,
    step: int,
    save_summary: bool = False,
    cluster_test_metrics: Optional[Dict[str, float]] = None,
    cluster_context_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Compute and log a variety of metrics."""
    model.eval()
    invariant_to = "both" if cfg.adapt.eval_s_from_zs is not None else "s"

    LOGGER.info("Encoding training set...")
    train = encode_dataset(
        cfg, data.train, model, recons=cfg.adapt.eval_on_recon, invariant_to=invariant_to
    )
    assert train.inv_s is not None
    if cfg.adapt.eval_on_recon:
        # don't encode test dataset
        test_repr = data.test
    else:
        test = encode_dataset(cfg, data.test, model, recons=False, invariant_to=invariant_to)
        assert test.inv_s is not None
        test_repr = test.inv_s

    LOGGER.info("\nComputing metrics...")
    evaluate(
        cfg=cfg,
        step=step,
        train_data=train.inv_s,
        test_data=test_repr,
        y_dim=data.y_dim,
        s_dim=data.s_dim,
        eval_on_recon=cfg.adapt.eval_on_recon,
        pred_s=False,
        save_summary=save_summary,
        cluster_test_metrics=cluster_test_metrics,
        cluster_context_metrics=cluster_context_metrics,
    )

    if cfg.adapt.eval_s_from_zs is not None:
        if cfg.adapt.eval_s_from_zs is EvalTrainData.train:
            assert train.inv_y is not None
            train_data = train.inv_y  # the part that is invariant to y corresponds to zs
        else:
            context = encode_dataset(
                cfg, data.context, model, recons=cfg.adapt.eval_on_recon, invariant_to="y"
            )
            assert context.inv_y is not None
            train_data = context.inv_y
        assert test.inv_y is not None, "if this test fails, you're evaluating on recons"
        evaluate(
            cfg=cfg,
            step=step,
            train_data=train_data,
            test_data=test.inv_y,
            y_dim=data.y_dim,
            s_dim=data.s_dim,
            name="s_from_zs",
            eval_on_recon=cfg.adapt.eval_on_recon,
            pred_s=True,
            save_summary=save_summary,
        )


def baseline_metrics(cfg: Config, data: DatasetTriplet) -> None:
    if isinstance(cfg.data, AdultConfig):
        LOGGER.info("Baselines...")
        train_data = data.train
        test_data = data.test
        if not isinstance(train_data, em.DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=False)

        for clf in [
            em.LR(),
            em.Majority(),
            em.Kamiran(classifier="LR"),
            em.LRCV(),
            em.SVM(),
        ]:
            preds = clf.run(train_data, test_data)
            compute_metrics(
                predictions=preds,
                actual=test_data,
                s_dim=data.s_dim,
                exp_name="original_data",
                model_name=clf.name,
                step=0,
                use_wandb=False,
            )


def fit_classifier(
    cfg: Config,
    input_shape: Sequence[int],
    train_data: DataLoader,
    train_on_recon: bool,
    pred_s: bool,
    target_dim: int,
    test_data: Optional[DataLoader] = None,
) -> Classifier:
    input_dim = input_shape[0]
    optimizer_kwargs = {"lr": cfg.adapt.eval_lr}
    clf_fn: ModelFn

    if train_on_recon:
        if isinstance(cfg.data, ImageDatasetConfig):
            if isinstance(cfg.data, CmnistConfig):
                clf_fn = Mp32x23Net(batch_norm=True)
            elif isinstance(cfg.data, CelebaConfig):
                clf_fn = Mp64x64Net(batch_norm=True)
            else:  # ISIC dataset

                def resnet50_ft(input_dim: int, target_dim: int) -> ResNet:
                    classifier = resnet50(pretrained=True)
                    classifier.fc = nn.Linear(classifier.fc.in_features, target_dim)
                    return classifier

                clf_fn = resnet50_ft
        else:

            def _adult_fc_net(input_dim: int, target_dim: int) -> nn.Sequential:
                encoder = FcNet(hidden_dims=[35])(input_dim=input_dim, target_dim=35)
                classifier = nn.Linear(35, target_dim)
                return nn.Sequential(encoder, classifier)

            optimizer_kwargs["weight_decay"] = 1e-8
            clf_fn = _adult_fc_net
    else:
        clf_fn = FcNet(hidden_dims=None)
        input_dim = prod(input_shape)
    clf_base = clf_fn(input_dim, target_dim=target_dim)

    num_classes = max(2, target_dim)
    clf: Classifier = Classifier(
        clf_base, num_classes=num_classes, optimizer_kwargs=optimizer_kwargs
    )
    clf.to(torch.device(cfg.misc.device))
    clf.fit(
        train_data,
        test_data=test_data,
        epochs=cfg.adapt.eval_epochs,
        device=torch.device(cfg.misc.device),
        pred_s=pred_s,
    )

    return clf


def evaluate(
    cfg: Config,
    step: int,
    train_data: ExtractableDataset,
    test_data: Dataset[Tuple[Tensor, Tensor, Tensor]],
    y_dim: int,
    s_dim: int,
    name: str = "",
    eval_on_recon: bool = True,
    pred_s: bool = False,
    save_summary: bool = False,
    cluster_test_metrics: Optional[Dict[str, float]] = None,
    cluster_context_metrics: Optional[Dict[str, float]] = None,
):
    input_shape = next(iter(train_data))[0].shape
    additional_entries = {}
    if cluster_test_metrics is not None:
        additional_entries.update({f"Clust/Test {k}": v for k, v in cluster_test_metrics.items()})
    if cluster_context_metrics is not None:
        additional_entries.update(
            {f"Clust/Context {k}": v for k, v in cluster_context_metrics.items()}
        )

    train_loader_kwargs = {}
    if cfg.adapt.balanced_eval:
        train_loader_kwargs["sampler"] = build_weighted_sampler_from_dataset(
            dataset=train_data,
            s_count=max(s_dim, 2),
            batch_size=cfg.adapt.eval_batch_size,
            oversample=cfg.adapt.oversample,
            balance_hierarchical=False,
        )
        train_loader_kwargs["shuffle"] = False  # the sampler shuffles for us
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.adapt.eval_batch_size,
        pin_memory=True,
        **train_loader_kwargs,
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.adapt.test_batch_size, shuffle=False, pin_memory=True
    )

    clf: Classifier = fit_classifier(
        cfg,
        input_shape,
        train_data=train_loader,
        train_on_recon=eval_on_recon,
        pred_s=pred_s,
        test_data=test_loader,
        target_dim=s_dim if pred_s else y_dim,
    )

    preds, labels, sens = clf.predict_dataset(test_loader, device=torch.device(cfg.misc.device))
    del train_loader  # try to prevent lock ups of the workers
    del test_loader
    preds = em.Prediction(hard=pd.Series(preds))
    if isinstance(cfg.data, CmnistConfig):
        sens_name = "colour"
    elif isinstance(cfg.data, CelebaConfig):
        sens_name = cfg.data.celeba_sens_attr.name
    elif isinstance(cfg.data, IsicConfig):
        sens_name = cfg.data.isic_sens_attr.name
    elif isinstance(cfg.data, AdultConfig):
        sens_name = str(adult.SENS_ATTRS[0])
    else:
        sens_name = "sens_Label"
    sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=[sens_name])
    labels_pd = pd.DataFrame(labels, columns=["labels"])
    actual = em.DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels_pd)
    compute_metrics(
        predictions=preds,
        actual=actual,
        exp_name=name,
        model_name="pytorch_classifier",
        step=step,
        s_dim=s_dim,
        save_summary=save_summary,
        use_wandb=cfg.misc.wandb is not WandbMode.disabled,
        additional_entries=additional_entries,
    )
    if isinstance(cfg.data, AdultConfig):
        train_data_tup, test_data_tup = get_data_tuples(train_data, test_data)

        train_data_tup, test_data_tup = make_tuple_from_data(
            train_data_tup, test_data_tup, pred_s=pred_s
        )
        for eth_clf in [em.LR(), em.LRCV()]:  # , em.LRCV(), em.SVM(kernel="linear")]:
            preds = eth_clf.run(train_data_tup, test_data_tup)
            compute_metrics(
                predictions=preds,
                actual=test_data_tup,
                exp_name=name,
                model_name=eth_clf.name,
                s_dim=s_dim,
                step=step,
                save_summary=save_summary,
                use_wandb=cfg.misc.wandb is not WandbMode.disabled,
                additional_entries=additional_entries,
            )


Invariance = Literal["s", "y", "both"]


def encode_dataset(
    cfg: Config, data: Dataset, generator: AutoEncoder, recons: bool, invariant_to: Invariance = "s"
) -> InvarianceDatasets:
    LOGGER.info("Encoding dataset...")
    all_inv_s = []
    all_inv_y = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=cfg.adapt.encode_batch_size, pin_memory=True, shuffle=False, num_workers=0
    )
    device = torch.device(cfg.misc.device)

    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            enc = generator.encode(x, stochastic=False)

            if invariant_to in ("s", "both"):
                all_inv_s.append(_get_classifer_input(cfg, enc, generator, recons, "s"))

            if invariant_to in ("y", "both"):
                all_inv_y.append(_get_classifer_input(cfg, enc, generator, recons, "y"))

    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    datasets: dict[Literal["inv_y", "inv_s"], TensorDataTupleDataset] = {}

    if all_inv_s:
        inv_s = torch.cat(all_inv_s, dim=0)
        datasets["inv_s"] = TensorDataTupleDataset(x=inv_s, s=all_s, y=all_y)

    if all_inv_y:
        inv_y = torch.cat(all_inv_y, dim=0)
        datasets["inv_y"] = TensorDataTupleDataset(x=inv_y, s=all_s, y=all_y)

    LOGGER.info("Done.")
    return InvarianceDatasets(inv_y=datasets.get("inv_y"), inv_s=datasets.get("inv_s"))


def _get_classifer_input(
    cfg: Config,
    enc: SplitEncoding,
    generator: AutoEncoder,
    recons: bool,
    invariance: Literal["s", "y"],
) -> Tensor:
    if recons:
        # `zs_m` has zs zeroed out
        if cfg.adapt.train_on_recon:
            zs_m, zy_m = generator.mask(enc, random=True)
        else:
            # if we didn't train with the random encodings, it probably doesn't make much
            # sense to evaluate with them; better to use null-sampling
            zs_m, zy_m = generator.mask(enc, random=False)
        z_m = zs_m if invariance == "s" else zy_m
        x_m = generator.decode(z_m, mode="hard")

        if isinstance(cfg.data, (CelebaConfig, IsicConfig)):
            x_m = 0.5 * x_m + 0.5
        if x_m.dim() > 2:
            x_m = x_m.clamp(min=0, max=1)
        classifier_input = x_m
    else:
        zs_m, zy_m = generator.mask(enc)
        # `zs_m` has zs zeroed out
        z_m = zs_m if invariance == "s" else zy_m
        classifier_input = generator.unsplit_encoding(z_m)
    return classifier_input.detach().cpu()
