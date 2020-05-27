from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from tqdm import tqdm

import wandb
from ethicml.algorithms.inprocess import LR
from ethicml.evaluators import run_metrics
from ethicml.metrics import TNR, TPR, Accuracy, ProbPos, RenyiCorrelation
from ethicml.utility import DataTuple, Prediction
from shared.data import DatasetTriplet, get_data_tuples
from shared.utils import wandb_log, prod
from clustering.configs import ClusterArgs
from clustering.models import Classifier, Encoder
from clustering.models.configs import fc_net, mp_32x32_net, mp_64x64_net

from .utils import log_images


def log_sample_images(args, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(args, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    args: ClusterArgs, model, data: DatasetTriplet, step: int, save_to_csv: Optional[Path] = None,
):
    """Compute and log a variety of metrics"""
    model.eval()
    print("Encoding training set...")
    train_inv_s = encode_dataset(
        args, data.train, model, recons=args.eval_on_recon, invariant_to="s"
    )
    if args.eval_on_recon:
        # don't encode test dataset
        test_repr = data.test
    else:
        test_repr = encode_dataset(args, data.test, model, recons=False, invariant_to="s")

    print("\nComputing metrics...")
    evaluate(
        args,
        step,
        train_inv_s,
        test_repr,
        name="x_rand_s",
        eval_on_recon=args.eval_on_recon,
        pred_s=False,
        save_to_csv=save_to_csv,
    )
    if args.three_way_split:
        print("Encoding training dataset (random y)...")
        train_rand_y = encode_dataset(
            args, data.train, model, recons=args.eval_on_recon, invariant_to="y"
        )
        evaluate(
            args,
            step,
            train_rand_y,
            test_repr,
            name="x_rand_y",
            eval_on_recon=args.eval_on_recon,
            pred_s=False,
            save_to_csv=save_to_csv,
        )


def compute_metrics(
    args: ClusterArgs, predictions: Prediction, actual, name: str, step: int, run_all=False
) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), TPR(), TNR(), RenyiCorrelation()],
            per_sens_metrics=[ProbPos(), TPR(), TNR()],
        )
        # logging_dict = {
        #     f"{name} Accuracy": metrics["Accuracy"],
        #     f"{name} TPR": metrics["TPR"],
        #     f"{name} TNR": metrics["TNR"],
        #     f"{name} PPV": metrics["PPV"],
        #     f"{name} P(Y=1|s=0)": metrics["prob_pos_sex_Male_0.0"],
        #     f"{name} P(Y=1|s=1)": metrics["prob_pos_sex_Male_1.0"],
        #     f"{name} P(Y=1|s=0) Ratio s0/s1": metrics["prob_pos_sex_Male_0.0/sex_Male_1.0"],
        #     f"{name} P(Y=1|s=0) Diff s0-s1": metrics["prob_pos_sex_Male_0.0-sex_Male_1.0"],
        #     f"{name} TPR|s=1": metrics["TPR_sex_Male_1.0"],
        #     f"{name} TPR|s=0": metrics["TPR_sex_Male_0.0"],
        #     f"{name} TPR Ratio s0/s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"],
        #     f"{name} TPR Diff s0-s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"],
        #     f"{name} PPV Ratio s0/s1": metrics["PPV_sex_Male_0.0/sex_Male_1.0"],
        #     f"{name} TNR Ratio s0/s1": metrics["TNR_sex_Male_0.0/sex_Male_1.0"],
        # }
        wandb_log(args, metrics, step=step)
    else:
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        wandb_log(args, {f"{name} Accuracy": metrics["Accuracy"]}, step=step)
    return metrics


def fit_classifier(
    args: ClusterArgs,
    input_shape: Sequence[int],
    train_data: DataLoader,
    train_on_recon: bool,
    pred_s: bool,
    test_data: Optional[DataLoader] = None,
):
    input_dim = input_shape[0]
    if args.dataset == "cmnist" and train_on_recon:
        clf_fn = mp_32x32_net
    elif args.dataset in ("celeba", "ssrp", "genfaces") and train_on_recon:
        clf_fn = mp_64x64_net
    else:
        clf_fn = fc_net
        input_dim = prod(input_shape)
    clf = clf_fn(input_dim, target_dim=args._y_dim)

    n_classes = args._y_dim if args._y_dim > 1 else 2
    clf: Classifier = Classifier(clf, num_classes=n_classes, optimizer_kwargs={"lr": args.eval_lr})
    clf.to(args._device)
    clf.fit(
        train_data, test_data=test_data, epochs=args.eval_epochs, device=args._device, pred_s=pred_s
    )

    return clf


def make_tuple_from_data(train, test, pred_s):
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return (DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y))


def evaluate(
    args: ClusterArgs,
    step: int,
    train_data: Dataset[Tuple[Tensor, Tensor, Tensor]],
    test_data: Dataset[Tuple[Tensor, Tensor, Tensor]],
    name: str,
    eval_on_recon: bool = True,
    pred_s: bool = False,
    save_to_csv: Optional[Path] = None,
):
    input_shape = next(iter(train_data))[0].shape

    if args.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_data, batch_size=args.test_batch_size, shuffle=False, pin_memory=True
        )

        clf: Classifier = fit_classifier(
            args,
            input_shape,
            train_data=train_loader,
            train_on_recon=eval_on_recon,
            pred_s=pred_s,
            test_data=test_loader,
        )

        preds, actual, sens = clf.predict_dataset(test_loader, device=args._device)
        preds = Prediction(hard=pd.Series(preds))
        sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=["sex_Male"])
        labels = pd.DataFrame(actual, columns=["labels"])
        actual = DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels)

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        clf = LR()
        preds = clf.run(train_data, test_data)
        actual = test_data

    full_name = f"{args.dataset}_{name}"
    full_name += "_s" if pred_s else "_y"
    full_name += "_on_recons" if eval_on_recon else "_on_encodings"
    metrics = compute_metrics(args, preds, actual, full_name, run_all=args._y_dim == 1, step=step)
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line

    if save_to_csv is not None and args.results_csv:
        assert isinstance(save_to_csv, Path)
        sweep_key = "Scale" if args.dataset == "cmnist" else "Mix_fact"
        sweep_value = str(args.scale) if args.dataset == "cmnist" else str(args.mixing_factor)
        results_path = save_to_csv / f"{full_name}_{args.results_csv}"
        value_list = ",".join([sweep_value] + [str(v) for v in metrics.values()])
        if not results_path.is_file():
            with results_path.open("w") as f:
                f.write(",".join([sweep_key] + [str(k) for k in metrics.keys()]) + "\n")  # header
                f.write(value_list + "\n")
        else:
            with results_path.open("a") as f:  # append to existing file
                f.write(value_list + "\n")
        print(f"Results have been written to {results_path.resolve()}")
        if args.use_wandb:
            for metric_name, value in metrics.items():
                wandb.run.summary[metric_name] = value

    return metrics, clf


def encode_dataset(
    args: ClusterArgs, data: Dataset, generator: Encoder,
) -> Dataset[Tuple[Tensor, Tensor, Tensor]]:
    print("Encoding dataset...", flush=True)  # flush to avoid conflict with tqdm
    all_enc = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=args.encode_batch_size, pin_memory=True, shuffle=False, num_workers=4
    )

    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(args._device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            enc = generator.encode(x, stochastic=False)
            all_enc.append(enc.detach().cpu())

    all_enc = torch.cat(all_enc, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_enc, all_s, all_y)
    print("Done.")

    return encoded_dataset
