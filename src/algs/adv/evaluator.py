from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Generic, Literal, TypeVar, overload

from conduit.data import TernarySample
from conduit.data.datasets import CdtDataLoader, CdtDataset
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from ranzen.misc import gcopy
import seaborn as sns
import torch
from torch import Tensor
from tqdm import tqdm
import umap
import wandb

from src.arch.predictors import Fcn
from src.data import DataModule, Dataset, group_id_to_label, labels_to_group_id, resolve_device
from src.evaluation.metrics import EmEvalPair, compute_metrics
from src.models import Classifier, OptimizerCfg, SplitLatentAe

__all__ = [
    "Evaluator",
    "InvariantDatasets",
    "encode_dataset",
    "visualize_clusters",
]


DY = TypeVar("DY", bound=Dataset[Tensor] | None)
DS = TypeVar("DS", bound=Dataset[Tensor] | None)


class EvalTrainData(Enum):
    """Dataset to use for training during evaluation."""

    train = auto()
    deployment = auto()


@dataclass(frozen=True)
class InvariantDatasets(Generic[DY, DS]):
    zs: DY
    zy: DS


# def log_sample_images(
#     *,
#     data: CdtVisionDataset[TernarySample[Tensor], Tensor, Tensor],
#     dm: DataModule,
#     name: str,
#     step: int,
#     num_samples: int = 64,
# ) -> None:
#     inds: list[int] = torch.randperm(len(data))[:num_samples].tolist()
#     images = data[inds]
#     log_images(images=images, dm=dm, name=f"Samples from {name}", prefix="eval", step=step)


InvariantAttr = Literal["zy", "zs", "both"]


_PBAR_COL: Final[str] = "#ffe252"


@overload
def encode_dataset(
    dl: CdtDataLoader[TernarySample[Tensor]],
    *,
    encoder: SplitLatentAe,
    device: str | torch.device,
    segment: Literal["zs"] = ...,
    use_amp: bool = False,
) -> InvariantDatasets[Dataset[Tensor], None]: ...


@overload
def encode_dataset(
    dl: CdtDataLoader[TernarySample[Tensor]],
    *,
    encoder: SplitLatentAe,
    device: str | torch.device,
    segment: Literal["zy"] = ...,
    use_amp: bool = False,
) -> InvariantDatasets[None, Dataset[Tensor]]: ...


@overload
def encode_dataset(
    dl: CdtDataLoader[TernarySample[Tensor]],
    *,
    encoder: SplitLatentAe,
    device: str | torch.device,
    segment: Literal["both"],
    use_amp: bool = False,
) -> InvariantDatasets[Dataset[Tensor], Dataset[Tensor]]: ...


def encode_dataset(
    dl: CdtDataLoader[TernarySample[Tensor]],
    *,
    encoder: SplitLatentAe,
    device: str | torch.device,
    segment: InvariantAttr = "zy",
    use_amp: bool = False,
) -> InvariantDatasets:
    device = resolve_device(device)
    zy_ls, zs_ls, s_ls, y_ls = [], [], [], []
    with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
        with torch.no_grad():
            for batch in tqdm(dl, desc="Encoding dataset", colour=_PBAR_COL):
                x = batch.x.to(device, non_blocking=True)
                s_ls.append(batch.s)
                y_ls.append(batch.y)

                # don't do the zs transform here because we might want to look at the raw distribution
                encodings = encoder.encode(x, transform_zs=False)

                if segment in ("zy", "both"):
                    zy_ls.append(encodings.zy.detach().cpu())

                if segment in ("zs", "both"):
                    zs_ls.append(encodings.zs.detach().cpu())

    s_ls = torch.cat(s_ls, dim=0)
    y_ls = torch.cat(y_ls, dim=0)
    zs_ds = None
    if zs_ls:
        zs_ds = CdtDataset(x=torch.cat(zs_ls, dim=0), s=s_ls, y=y_ls)

    zy_ds = None
    if zy_ls:
        zy_ds = CdtDataset(x=torch.cat(zy_ls, dim=0), s=s_ls, y=y_ls)

    logger.info("Finished encoding")

    return InvariantDatasets(zs=zs_ds, zy=zy_ds)


def _log_enc_statistics(encoded: Dataset[Tensor], *, step: int | None, s_count: int) -> None:
    """Compute and log statistics about the encoding."""
    x, y, s = encoded.x, encoded.y, encoded.s
    class_ids = labels_to_group_id(s=s, y=y, s_count=s_count)

    logger.info("Starting UMAP...")
    mapper = umap.UMAP(n_neighbors=25, n_components=2)  # type: ignore
    umap_z = mapper.fit_transform(x.numpy())
    umap_plot = visualize_clusters(umap_z, labels=class_ids, s_count=s_count)
    to_log: dict[str, wandb.Image | float] = {"umap": wandb.Image(umap_plot)}
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


@dataclass
class Evaluator:
    steps: int = 10_000
    batch_size: int = 128
    eval_s_from_zs: EvalTrainData | None = None
    balanced_sampling: bool = True
    umap_viz: bool = False
    save_summary: bool = True
    model: Fcn = field(default_factory=Fcn)
    opt: OptimizerCfg = field(default_factory=OptimizerCfg)
    """Optimization parameters."""

    def _fit_classifier(
        self, dm: DataModule, *, pred_s: bool, input_dim: int, device: torch.device
    ) -> Classifier:
        model, _ = self.model(input_dim, target_dim=dm.card_y)

        clf = Classifier(model, opt=self.opt)

        train_dl = dm.train_dataloader(batch_size=self.batch_size, balance=self.balanced_sampling)

        clf.to(torch.device(device))
        clf.fit(
            train_data=train_dl,
            test_data=None,
            steps=self.steps,
            device=torch.device(device),
            pred_s=pred_s,
            use_wandb=False,
            val_interval=1.1,  # never
        )

        return clf

    def _evaluate(
        self,
        dm: DataModule,
        *,
        input_dim: int,
        device: torch.device,
        step: int | None = None,
        name: str = "",
        pred_s: bool = False,
    ) -> None:
        clf = self._fit_classifier(dm=dm, pred_s=False, device=device, input_dim=input_dim)
        et = clf.predict(dm.test_dataloader(), device=torch.device(device))
        pair = EmEvalPair.from_et(et=et, pred_s=pred_s)
        compute_metrics(
            pair=pair,
            exp_name=name,
            step=step,
            save_summary=self.save_summary,
            prefix="test",
            use_wandb=True,
        )

    def run(
        self,
        dm: DataModule,
        *,
        encoder: SplitLatentAe,
        device: str | torch.device | int,
        step: int | None = None,
    ) -> DataModule:
        device = resolve_device(device)
        encoder.eval()
        segment = "both" if self.eval_s_from_zs is not None else "zy"

        logger.info("Encoding training set")
        train_eval = encode_dataset(
            dl=dm.train_dataloader(eval=True, batch_size=dm.batch_size_te),
            encoder=encoder,
            device=device,
            segment=segment,
        )
        logger.info("Encoding test set")
        test_eval = encode_dataset(
            dl=dm.test_dataloader(), encoder=encoder, device=device, segment=segment
        )

        s_count = dm.dim_s if dm.dim_s > 1 else 2
        if self.umap_viz:
            _log_enc_statistics(test_eval.zy, step=step, s_count=s_count)
        if test_eval.zs is not None and (test_eval.zs.x[0].size(1) == 1):
            zs = test_eval.zs.x[:, 0].view((test_eval.zs.x.size(0),)).sigmoid()
            zs_np = zs.detach().cpu().numpy()
            fig, plot = plt.subplots(dpi=200, figsize=(6, 4))
            plot.hist(zs_np, bins=20, range=(0, 1))
            plot.set_xlim(left=0, right=1)
            fig.tight_layout()
            wandb.log({"zs_histogram": wandb.Image(fig)}, step=step)

        enc_size = encoder.encoding_size
        dm_zy = gcopy(dm, deep=False, train=train_eval.zy, test=test_eval.zy)
        logger.info("\nComputing metrics...")
        self._evaluate(
            dm=dm_zy,
            device=device,
            step=step,
            name="y_from_zy",
            pred_s=False,
            input_dim=enc_size.zy,
        )

        if self.eval_s_from_zs is not None:
            if self.eval_s_from_zs is EvalTrainData.train:
                train_data = train_eval.zs  # the part that is invariant to y corresponds to zs
            else:
                encoded_dep = encode_dataset(
                    dl=dm.deployment_dataloader(eval=True),
                    encoder=encoder,
                    device=device,
                    segment="zs",
                )
                train_data = encoded_dep.zs
            dm_zs = gcopy(dm, deep=False, train=train_data, test=test_eval.zs)
            self._evaluate(
                dm=dm_zs,
                device=device,
                step=step,
                name="s_from_zs",
                pred_s=True,
                input_dim=enc_size.zs,
            )
        return dm_zy

    def __call__(
        self,
        dm: DataModule,
        *,
        encoder: SplitLatentAe,
        device: str | torch.device | int,
        step: int | None = None,
    ) -> DataModule:
        return self.run(dm=dm, encoder=encoder, device=device, step=step)
