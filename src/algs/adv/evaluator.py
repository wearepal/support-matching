from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar, overload
from typing_extensions import Literal

from conduit.data import CdtDataLoader
from conduit.data.datasets.base import CdtDataset
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample
import ethicml as em
from ethicml.utility.data_structures import LabelTuple
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from ranzen.misc import gcopy
import seaborn as sns
import torch
from torch import Tensor
from tqdm import tqdm
import umap
import wandb

from src.arch.common import Activation
from src.arch.predictors import Fcn
from src.configs.enums import EvalTrainData
from src.data import (
    DataModule,
    Dataset,
    group_id_to_label,
    labels_to_group_id,
    resolve_device,
)
from src.evaluation.metrics import compute_metrics
from src.logging import log_images
from src.models import AutoEncoder, Classifier, SplitEncoding

__all__ = [
    "Evaluator",
    "InvariantDatasets",
    "encode_dataset",
    "log_sample_images",
    "visualize_clusters",
]


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


def _get_classifer_input(
    dm: DataModule,
    *,
    encodings: SplitEncoding,
    encoder: AutoEncoder,
    invariant_to: Literal["s", "y"],
) -> Tensor:
    zs_m, zy_m = encodings.mask()
    # `zs_m` has zs zeroed out
    z_m = zs_m if invariant_to == "s" else zy_m
    classifier_input = z_m.join()
    return classifier_input.detach().cpu()


InvariantAttr = Literal["s", "y", "both"]


@overload
def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
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
    device: str | torch.device,
    invariant_to: Literal["both"],
) -> InvariantDatasets[Dataset, Dataset]:
    ...


def encode_dataset(
    *,
    dm: DataModule,
    dl: CdtDataLoader[TernarySample],
    encoder: AutoEncoder,
    device: str | torch.device,
    invariant_to: InvariantAttr = "s",
) -> InvariantDatasets:
    logger.info("Encoding dataset")
    all_inv_s = []
    all_inv_y = []
    all_s = []
    all_y = []

    device = torch.device(device)

    with torch.no_grad():
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
                        invariant_to="s",
                    )
                )

            if invariant_to in ("y", "both"):
                all_inv_y.append(
                    _get_classifer_input(
                        dm=dm,
                        encodings=encodings,
                        encoder=encoder,
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

    logger.info("Done.")
    return InvariantDatasets(inv_y=inv_y, inv_s=inv_s)


def _log_enc_statistics(encoded: Dataset, *, step: int | None, s_count: int) -> None:
    """Compute and log statistics about the encoding."""
    x, y, s = encoded.x, encoded.y, encoded.s
    class_ids = labels_to_group_id(s=s, y=y, s_count=s_count)

    logger.info("Starting UMAP...")
    mapper = umap.UMAP(n_neighbors=25, n_components=2)  # type: ignore
    umap_z = mapper.fit_transform(x.numpy())
    umap_plot = visualize_clusters(umap_z, labels=class_ids, s_count=s_count)
    to_log = {"umap": wandb.Image(umap_plot)}
    logger.info("Done.")

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
    sns.despine(left=True, bottom=True, right=True)  # type: ignore

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


@dataclass
class Evaluator:
    steps: int = 10_000
    batch_size_tr: int = 128
    lr: float = 1.0e-4
    hidden_dim: Optional[int] = None
    num_hidden: int = 0
    eval_s_from_zs: Optional[EvalTrainData] = None
    balanced_sampling: bool = True
    umap_viz: bool = False
    save_summary: bool = True

    def _fit_classifier(
        self,
        dm: DataModule,
        *,
        pred_s: bool,
        device: torch.device,
    ) -> Classifier:
        input_dim = dm.dim_x[0]
        optimizer_kwargs = {"lr": self.lr}
        clf_fn = Fcn(
            hidden_dim=self.hidden_dim, num_hidden=self.num_hidden, activation=Activation.GELU
        )
        input_dim = np.product(dm.dim_x)
        clf_base, _ = clf_fn(input_dim, target_dim=dm.card_y)

        clf = Classifier(clf_base, optimizer_kwargs=optimizer_kwargs)

        train_dl = dm.train_dataloader(
            batch_size=self.batch_size_tr, balance=self.balanced_sampling
        )
        test_dl = dm.test_dataloader()

        clf.to(torch.device(device))
        clf.fit(
            train_data=train_dl,
            test_data=test_dl,
            steps=self.steps,
            device=torch.device(device),
            pred_s=pred_s,
        )

        return clf

    def _evaluate(
        self,
        dm: DataModule,
        *,
        device: torch.device,
        step: int | None = None,
        name: str = "",
        pred_s: bool = False,
    ) -> None:
        clf = self._fit_classifier(dm=dm, pred_s=False, device=device)

        # TODO: the soft predictions should only be computed if they're needed
        preds, labels, sens, _ = clf.predict_dataset(
            dm.test_dataloader(), device=torch.device(device), with_soft=True
        )
        # TODO: investigate why the histogram plotting fails when s_dim != 1
        # if (cfg.logging.mode is not WandbMode.disabled) and (dm.card_s == 2):
        #     plot_histogram_by_source(soft_preds, s=sens, y=labels, step=step, name=name)
        preds = em.Prediction(hard=pd.Series(preds))
        sens_pd = pd.Series(sens.numpy().astype(np.float32), name="subgroup")
        labels_pd = pd.Series(labels.cpu().numpy(), name="labels")
        actual = LabelTuple.from_df(s=sens_pd, y=sens_pd if pred_s else labels_pd)
        compute_metrics(
            predictions=preds,
            actual=actual,
            exp_name=name,
            model_name="pytorch_classifier",
            step=step,
            s_dim=dm.card_s,
            save_summary=self.save_summary,
            prefix="test",
        )

    def run(
        self,
        dm: DataModule,
        *,
        encoder: AutoEncoder,
        device: str | torch.device | int,
        step: int | None = None,
    ) -> None:
        device = resolve_device(device)
        encoder.eval()
        invariant_to = "both" if self.eval_s_from_zs is not None else "s"

        logger.info("Encoding training set...")
        train_eval = encode_dataset(
            dm=dm,
            dl=dm.train_dataloader(eval=True, batch_size=dm.batch_size_te),
            encoder=encoder,
            device=device,
            invariant_to=invariant_to,
        )
        test_eval = encode_dataset(
            dm=dm,
            dl=dm.test_dataloader(),
            encoder=encoder,
            device=device,
            invariant_to=invariant_to,
        )

        s_count = dm.dim_s if dm.dim_s > 1 else 2
        if self.umap_viz:
            _log_enc_statistics(test_eval.inv_s, step=step, s_count=s_count)
        if test_eval.inv_y is not None and (test_eval.inv_y.x[0].size(1) == 1):
            zs = test_eval.inv_y.x[:, 0].view((test_eval.inv_y.x.size(0),)).sigmoid()
            zs_np = zs.detach().cpu().numpy()
            fig, plot = plt.subplots(dpi=200, figsize=(6, 4))
            plot.hist(zs_np, bins=20, range=(0, 1))
            plot.set_xlim(left=0, right=1)
            fig.tight_layout()
            wandb.log({"zs_histogram": wandb.Image(fig)}, step=step)

        dm_cp = gcopy(dm, deep=False, train=train_eval.inv_s, test=test_eval.inv_s)
        logger.info("\nComputing metrics...")
        self._evaluate(dm=dm_cp, device=device, step=step, name="y_from_zy", pred_s=False)

        if self.eval_s_from_zs is not None:
            if self.eval_s_from_zs is EvalTrainData.train:
                train_data = train_eval.inv_y  # the part that is invariant to y corresponds to zs
            else:
                encoded_dep = encode_dataset(
                    dm=dm,
                    dl=dm.deployment_dataloader(eval=True, num_workers=0),
                    encoder=encoder,
                    device=device,
                    invariant_to="y",
                )
                train_data = encoded_dep.inv_y
            dm_cp = gcopy(dm, deep=False, train=train_data, test=test_eval.inv_y)
            self._evaluate(dm=dm_cp, device=device, step=step, name="s_from_zs", pred_s=True)

    def __call__(
        self,
        dm: DataModule,
        *,
        encoder: AutoEncoder,
        device: str | torch.device | int,
        step: int | None = None,
    ) -> None:
        return self.run(dm=dm, encoder=encoder, device=device, step=step)
