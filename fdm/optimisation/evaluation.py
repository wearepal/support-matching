import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import ethicml as em
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing_extensions import Literal

from fdm.models import AutoEncoder, Classifier
from shared.configs import DS, Config
from shared.data import DatasetTriplet, get_data_tuples
from shared.models.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import ModelFn, compute_metrics, make_tuple_from_data, prod

from .utils import log_images

log = logging.getLogger(__name__.split(".")[-1].upper())


def log_sample_images(cfg: Config, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(cfg, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    cfg: Config,
    model,
    data: DatasetTriplet,
    step: int,
    save_to_csv: Optional[Path] = None,
    cluster_test_metrics: Optional[Dict[str, float]] = None,
    cluster_context_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Compute and log a variety of metrics."""
    model.eval()

    log.info("Encoding training set...")
    train_inv_s = encode_dataset(
        cfg, data.train, model, recons=cfg.fdm.eval_on_recon, invariant_to="s"
    )
    if cfg.fdm.eval_on_recon:
        # don't encode test dataset
        test_repr = data.test
    else:
        test_repr = encode_dataset(cfg, data.test, model, recons=False, invariant_to="s")

    log.info("\nComputing metrics...")
    evaluate(
        cfg,
        step,
        train_inv_s,
        test_repr,
        name="x_zero_s",
        eval_on_recon=cfg.fdm.eval_on_recon,
        pred_s=False,
        save_to_csv=save_to_csv,
        cluster_test_metrics=cluster_test_metrics,
        cluster_context_metrics=cluster_context_metrics,
    )


def baseline_metrics(cfg: Config, data: DatasetTriplet, save_to_csv: Optional[Path]) -> None:
    if cfg.data.dataset not in (DS.cmnist, DS.celeba, DS.genfaces):
        log.info("Baselines...")
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
                cfg=cfg,
                predictions=preds,
                actual=test_data,
                exp_name="original_data",
                model_name=clf.name,
                step=0,
                save_to_csv=save_to_csv,
                results_csv=cfg.misc.results_csv,
                use_wandb=False,
            )


def fit_classifier(
    cfg: Config,
    input_shape: Sequence[int],
    train_data: DataLoader,
    train_on_recon: bool,
    pred_s: bool,
    test_data: Optional[DataLoader] = None,
) -> Classifier:
    input_dim = input_shape[0]
    clf_fn: ModelFn
    if cfg.data.dataset == DS.cmnist and train_on_recon:
        clf_fn = Mp32x23Net(batch_norm=True)
    elif cfg.data.dataset in (DS.celeba, DS.genfaces) and train_on_recon:
        clf_fn = Mp64x64Net(batch_norm=True)
    else:
        clf_fn = FcNet(hidden_dims=None)
        input_dim = prod(input_shape)
    clf = clf_fn(input_dim, target_dim=cfg.misc._y_dim)

    n_classes = cfg.misc._y_dim if cfg.misc._y_dim > 1 else 2
    clf: Classifier = Classifier(
        clf, num_classes=n_classes, optimizer_kwargs={"lr": cfg.fdm.eval_lr}
    )
    clf.to(cfg.misc._device)
    clf.fit(
        train_data,
        test_data=test_data,
        epochs=cfg.fdm.eval_epochs,
        device=cfg.misc._device,
        pred_s=pred_s,
    )

    return clf


def evaluate(
    cfg: Config,
    step: int,
    train_data: "Dataset[Tuple[Tensor, Tensor, Tensor]]",
    test_data: "Dataset[Tuple[Tensor, Tensor, Tensor]]",
    name: str,
    eval_on_recon: bool = True,
    pred_s: bool = False,
    save_to_csv: Optional[Path] = None,
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

    if cfg.data.dataset in (DS.cmnist, DS.celeba, DS.genfaces):

        train_loader = DataLoader(
            train_data, batch_size=cfg.fdm.batch_size, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_data, batch_size=cfg.fdm.test_batch_size, shuffle=False, pin_memory=True
        )

        clf: Classifier = fit_classifier(
            cfg,
            input_shape,
            train_data=train_loader,
            train_on_recon=eval_on_recon,
            pred_s=pred_s,
            test_data=test_loader,
        )

        preds, labels, sens = clf.predict_dataset(
            test_loader, device=torch.device(cfg.misc._device)
        )
        preds = em.Prediction(hard=pd.Series(preds))
        if cfg.data.dataset == DS.cmnist:
            sens_name = "colour"
        elif cfg.data.dataset == DS.celeba:
            sens_name = cfg.data.celeba_sens_attr
        else:
            sens_name = "sens_Label"
        sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=[sens_name])
        labels_pd = pd.DataFrame(labels, columns=["labels"])
        actual = em.DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels_pd)
        compute_metrics(
            cfg,
            preds,
            actual,
            name,
            "pytorch_classifier",
            step=step,
            save_to_csv=save_to_csv,
            results_csv=cfg.misc.results_csv,
            use_wandb=cfg.misc.use_wandb,
            additional_entries=additional_entries,
        )
    else:
        if not isinstance(train_data, em.DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        for eth_clf in [em.LR(), em.LRCV()]:  # , em.LRCV(), em.SVM(kernel="linear")]:
            preds = eth_clf.run(train_data, test_data)
            compute_metrics(
                cfg,
                preds,
                test_data,
                name,
                eth_clf.name,
                step=step,
                save_to_csv=save_to_csv,
                results_csv=cfg.misc.results_csv,
                use_wandb=cfg.misc.use_wandb,
                additional_entries=additional_entries,
            )


def encode_dataset(
    cfg: Config,
    data: Dataset,
    generator: AutoEncoder,
    recons: bool,
    invariant_to: Literal["s", "y"] = "s",
) -> "TensorDataset":
    log.info("Encoding dataset...")
    all_x_m = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=cfg.fdm.encode_batch_size, pin_memory=True, shuffle=False, num_workers=0
    )

    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(cfg.misc._device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            enc = generator.encode(x, stochastic=False)
            if recons:
                if cfg.fdm.train_on_recon:
                    zs_m, zy_m = generator.mask(enc, random=True)
                else:
                    # if we didn't train with the random encodings, it probably doesn't make much
                    # sense to evaluate with them; better to use null-sampling
                    zs_m, zy_m = generator.mask(enc, random=False)
                z_m = zs_m if invariant_to == "s" else zy_m
                x_m = generator.decode(z_m, mode="hard")

                if cfg.data.dataset in (DS.celeba, DS.genfaces):
                    x_m = 0.5 * x_m + 0.5
                if x.dim() > 2:
                    x_m = x_m.clamp(min=0, max=1)
            else:
                zs_m, zy_m = generator.mask(enc)
                # `zs_m` has zs zeroed out
                x_m = zs_m if invariant_to == "s" else zy_m

            all_x_m.append(x_m.detach().cpu())

    all_x_m = torch.cat(all_x_m, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_x_m, all_s, all_y)
    log.info("Done.")

    return encoded_dataset
