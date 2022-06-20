from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    overload,
)
from typing_extensions import Literal

from conduit.data import CdtDataLoader
from conduit.data.datasets.base import CdtDataset
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.datasets.vision.celeba import CelebA
from conduit.data.datasets.vision.cmnist import ColoredMNIST
from conduit.data.structures import TernarySample
import ethicml as em
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from ranzen.misc import gcopy
import seaborn as sns
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
from tqdm import tqdm
import umap
import wandb

from shared.configs import Config, EvalTrainData, WandbMode
from shared.data.data_module import DataModule, Dataset
from shared.data.utils import group_id_to_label, labels_to_group_id
from shared.models.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import compute_metrics, plot_histogram_by_source, prod

from .utils import log_images

if TYPE_CHECKING:
    from advrep.models.autoencoder import AutoEncoder, SplitEncoding
    from advrep.models.classifier import Classifier

__all__ = [
    "InvariantDatasets",
    "encode_dataset",
    "evaluate",
    "fit_classifier",
    "log_metrics",
    "log_sample_images",
    "visualize_clusters",
]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


DY = TypeVar("DY", bound=Optional[Dataset])
DS = TypeVar("DS", bound=Optional[Dataset])


@dataclass(frozen=True)
class InvariantDatasets(Generic[DY, DS]):
    inv_y: DY
    inv_s: DS


def log_sample_images(
    *,
    data: CdtVisionDataset,
    dm: DataModule,
    name: str,
    step: int,
    num_samples: int = 64,
) -> None:
    inds = torch.randperm(len(data))[:num_samples]
    images = data[inds.tolist()]
    log_images(images=images, dm=dm, name=f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    cfg: Config,
    *,
    dm: DataModule,
    encoder: AutoEncoder,
    step: int,
    device: str | torch.device,
    save_summary: bool = False,
    cluster_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Compute and log a variety of metrics."""
    encoder.eval()
    invariant_to = "both" if cfg.alg.eval_s_from_zs is not None else "s"

    LOGGER.info("Encoding training set...")
    train_eval = encode_dataset(
        dm=dm,
        dl=dm.train_dataloader(eval=True, num_workers=0),
        encoder=encoder,
        recons=cfg.alg.eval_on_recon,
        device=device,
        invariant_to=invariant_to,
    )
    if cfg.alg.eval_on_recon:
        # don't encode test dataset
        test_eval = dm.test
    else:
        test_eval = encode_dataset(
            dm=dm,
            dl=dm.test_dataloader(num_workers=0),
            encoder=encoder,
            recons=False,
            device=device,
            invariant_to=invariant_to,
        )

        if cfg.logging.mode is not WandbMode.disabled:
            s_count = dm.dim_s if dm.dim_s > 1 else 2
            if cfg.logging.umap:
                _log_enc_statistics(test_eval.inv_s, step=step, s_count=s_count)
            if test_eval.inv_y is not None and cfg.alg.zs_dim == 1:
                zs = test_eval.inv_y.x[:, 0].view((test_eval.inv_y.x.size(0),)).sigmoid()
                zs_np = zs.detach().cpu().numpy()
                fig, plot = plt.subplots(dpi=200, figsize=(6, 4))
                plot.hist(zs_np, bins=20, range=(0, 1))
                plot.set_xlim(left=0, right=1)
                fig.tight_layout()
                wandb.log({"zs_histogram": wandb.Image(fig)}, step=step)

    dm_cp = gcopy(dm, deep=False, train=train_eval.inv_s, test=test_eval.inv_s)
    LOGGER.info("\nComputing metrics...")
    evaluate(
        cfg=cfg,
        dm=dm_cp,
        step=step,
        eval_on_recon=cfg.alg.eval_on_recon,
        pred_s=False,
        save_summary=save_summary,
        device=device,
        cluster_metrics=cluster_metrics,
    )

    if cfg.alg.eval_s_from_zs is not None:
        if cfg.alg.eval_s_from_zs is EvalTrainData.train:
            train_data = train_eval.inv_y  # the part that is invariant to y corresponds to zs
        else:
            encoded_dep = encode_dataset(
                dm=dm,
                dl=dm.deployment_dataloader(eval=True, num_workers=0),
                encoder=encoder,
                recons=cfg.alg.eval_on_recon,
                device=device,
                invariant_to="y",
            )
            train_data = encoded_dep.inv_y
        dm_cp = gcopy(dm, deep=False, train=train_data, test=test_eval.inv_y)
        evaluate(
            cfg=cfg,
            dm=dm,
            step=step,
            name="s_from_zs",
            device=device,
            eval_on_recon=cfg.alg.eval_on_recon,
            pred_s=True,
            save_summary=save_summary,
        )


def _log_enc_statistics(encoded: Dataset, *, step: int, s_count: int):
    """Compute and log statistics about the encoding."""
    x, y, s = encoded.x, encoded.y, encoded.s
    class_ids = labels_to_group_id(s=s, y=y, s_count=s_count)

    LOGGER.info("Starting UMAP...")
    mapper = umap.UMAP(n_neighbors=25, n_components=2)  # type: ignore
    umap_z = mapper.fit_transform(x.numpy())
    umap_plot = visualize_clusters(umap_z, labels=class_ids, s_count=s_count)
    to_log = {"umap": wandb.Image(umap_plot)}
    LOGGER.info("Done.")

    for y_value in y.unique():
        for s_value in s.unique():
            mask = (y == y_value) & (s == s_value)
            to_log[f"enc_mean_s={s_value}_y={y_value}"] = x[mask].mean().item()
    wandb.log(to_log, step=step)


def visualize_clusters(
    x: np.ndarray | Tensor,
    *,
    labels: np.ndarray | Tensor,
    s_count: int,
    title: str | None = None,
    legend: bool = True,
) -> plt.Figure:  # type: ignore
    if x.shape[1] != 2:
        raise ValueError("Cluster-visualization can only be performed for 2-dimensional inputs.")
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels_ls = labels.detach().cpu().long().tolist()
    else:
        labels_ls = labels.astype("uint").tolist()

    classes = np.unique(labels)
    num_classes = len(classes)
    fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
    cmap = ListedColormap(sns.color_palette("bright", num_classes).as_hex())  # type: ignore
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=labels_ls, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True, right=True)

    if legend:

        def _flip(items: Sequence[Any], ncol: int) -> Sequence[Any]:
            # return itertools.chain(*[items[i::ncol] for i in range(ncol)])
            return items

        plt.legend(
            handles=_flip(sc.legend_elements()[0], 5),
            labels=_flip(
                [
                    f"s={group_id_to_label(class_id, s_count=s_count, label='s')},"
                    f"y={group_id_to_label(class_id, s_count=s_count, label='y')}"
                    for class_id in classes.tolist()
                ],
                5,
            ),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.3, -0.03),
            ncol=4,
        )

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def fit_classifier(
    cfg: Config,
    *,
    dm: DataModule,
    train_on_recon: bool,
    pred_s: bool,
    device: str | torch.device,
) -> Classifier:
    input_dim = dm.dim_x[0]
    optimizer_kwargs = {"lr": cfg.alg.eval_lr}

    if train_on_recon:
        if isinstance(dm.train, CdtVisionDataset):
            if isinstance(dm.train, ColoredMNIST):
                clf_fn = Mp32x23Net(batch_norm=True)
            elif isinstance(dm.train, CelebA):
                clf_fn = Mp64x64Net(batch_norm=True)
            else:

                def resnet50_ft(input_dim: int, *, target_dim: int) -> ResNet:
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
        clf_fn = FcNet(hidden_dims=cfg.alg.eval_hidden_dims)
        input_dim = prod(dm.dim_x)
    clf_base = clf_fn(input_dim, target_dim=dm.card_y)

    from advrep.models.classifier import Classifier

    clf = Classifier(clf_base, optimizer_kwargs=optimizer_kwargs)

    train_dl = dm.train_dataloader(
        batch_size=cfg.alg.eval_batch_size, balance=cfg.alg.balanced_eval
    )
    test_dl = dm.test_dataloader()

    clf.to(torch.device(device))
    clf.fit(
        train_data=train_dl,
        test_data=test_dl,
        epochs=cfg.alg.eval_epochs,
        device=torch.device(device),
        pred_s=pred_s,
    )

    return clf


def evaluate(
    cfg: Config,
    *,
    dm: DataModule,
    step: int,
    device: str | torch.device,
    name: str = "",
    eval_on_recon: bool = True,
    pred_s: bool = False,
    save_summary: bool = False,
    cluster_metrics: Optional[Dict[str, float]] = None,
) -> None:

    clf = fit_classifier(cfg, dm=dm, train_on_recon=eval_on_recon, pred_s=pred_s, device=device)

    # TODO: the soft predictions should only be computed if they're needed
    preds, labels, sens, soft_preds = clf.predict_dataset(
        dm.test_dataloader(), device=torch.device(device), with_soft=True
    )
    # TODO: investigate why the histogram plotting fails when s_dim != 1
    if (cfg.logging.mode is not WandbMode.disabled) and (dm.card_s == 2):
        plot_histogram_by_source(soft_preds, s=sens, y=labels, step=step, name=name)
    preds = em.Prediction(hard=pd.Series(preds))
    sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=["subgroup"])
    labels_pd = pd.DataFrame(labels.cpu().numpy(), columns=["labels"])
    actual = em.DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels_pd)
    compute_metrics(
        predictions=preds,
        actual=actual,
        exp_name=name,
        model_name="pytorch_classifier",
        step=step,
        s_dim=dm.card_s,
        save_summary=save_summary,
        use_wandb=(cfg.logging.mode is not WandbMode.disabled),
        additional_entries=cluster_metrics,
        prefix="test",
    )


InvariantAttr = Literal["s", "y", "both"]


@overload
def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
    recons: bool,
    device: str | torch.device,
    invariant_to: Literal["y"] = ...,
) -> InvariantDatasets[Dataset, None]:
    ...


@overload
def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
    recons: bool,
    device: str | torch.device,
    invariant_to: Literal["s"] = ...,
) -> InvariantDatasets[None, Dataset]:
    ...


@overload
def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
    recons: bool,
    device: str | torch.device,
    invariant_to: Literal["both"],
) -> InvariantDatasets[Dataset, Dataset]:
    ...


def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
    recons: bool,
    device: str | torch.device,
    invariant_to: InvariantAttr = "s",
) -> InvariantDatasets:
    LOGGER.info("Encoding dataset...")
    all_inv_s = []
    all_inv_y = []
    all_s = []
    all_y = []

    device = torch.device(device)

    with torch.set_grad_enabled(False):
        for batch in tqdm(dl):

            x = batch.x.to(device, non_blocking=True)
            all_s.append(batch.s)
            all_y.append(batch.y)

            # don't do the zs transform here because we might want to look at the raw distribution
            encodings = encoder.encode(x, transform_zs=False)

            if invariant_to in ("s", "both"):
                all_inv_s.append(
                    _get_classifer_input(
                        dm=dm,
                        encodings=encodings,
                        encoder=encoder,
                        recons=recons,
                        invariant_to="s",
                    )
                )

            if invariant_to in ("y", "both"):
                all_inv_y.append(
                    _get_classifer_input(
                        dm=dm,
                        encodings=encodings,
                        encoder=encoder,
                        recons=recons,
                        invariant_to="y",
                    )
                )

    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    inv_y = None
    if all_inv_y:
        inv_y = torch.cat(all_inv_y, dim=0)
        inv_y = CdtDataset(x=inv_y, s=all_s, y=all_y)

    inv_s = None
    if all_inv_s:
        inv_s = torch.cat(all_inv_s, dim=0)
        inv_s = CdtDataset(x=inv_s, s=all_s, y=all_y)

    LOGGER.info("Done.")
    return InvariantDatasets(inv_y=inv_y, inv_s=inv_s)


def _get_classifer_input(
    dm: DataModule,
    *,
    encodings: SplitEncoding,
    encoder: AutoEncoder,
    recons: bool,
    invariant_to: Literal["s", "y"],
) -> Tensor:
    if recons:
        # `zs_m` has zs zeroed out
        # if we didn't train with the random encodings, it probably doesn't make much
        # sense to evaluate with them; better to use null-sampling
        zs_m, zy_m = encodings.mask(random=False)
        z_m = zs_m if invariant_to == "s" else zy_m
        x_m = encoder.decode(z_m, mode="hard")

        x_m = dm.denormalize(x_m)
        if x_m.dim() > 2:
            x_m = x_m.clamp(min=0, max=1)
        classifier_input = x_m
    else:
        zs_m, zy_m = encodings.mask()
        # `zs_m` has zs zeroed out
        z_m = zs_m if invariant_to == "s" else zy_m
        classifier_input = z_m.join()
    return classifier_input.detach().cpu()
