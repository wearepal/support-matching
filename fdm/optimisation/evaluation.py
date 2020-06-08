from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from tqdm import tqdm

import wandb
from ethicml.algorithms import inprocess as algos
from ethicml.evaluators import run_metrics
from ethicml.metrics import TNR, TPR, Accuracy, ProbPos, RenyiCorrelation
from ethicml.utility import DataTuple, Prediction
from shared.configs import BaseArgs
from shared.data import DatasetTriplet, get_data_tuples
from shared.models.configs.classifiers import mp_32x32_net, fc_net, mp_64x64_net
from shared.utils import wandb_log, prod
from fdm.configs import VaeArgs
from fdm.models import Classifier, AutoEncoder

from .utils import log_images

__all__ = ["compute_metrics", "make_tuple_from_data"]


def log_sample_images(args, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(args, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    args: VaeArgs,
    model,
    data: DatasetTriplet,
    step: int,
    save_to_csv: Optional[Path] = None,
    run_baselines: bool = False,
):
    """Compute and log a variety of metrics."""
    model.eval()
    if run_baselines:
        print("Baselines...")
        baseline_metrics(args, data, step)

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


def baseline_metrics(args: VaeArgs, data: DatasetTriplet, step: int) -> None:
    if args.dataset not in ("cmnist", "celeba", "ssrp", "genfaces"):
        train_data = data.train
        test_data = data.test
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=False)

        for clf in [algos.Majority(), algos.Kamiran(classifier="LR"), algos.LRCV()]:
            preds = clf.run(train_data, test_data)
            compute_metrics(args, preds, test_data, "baseline", clf.name, step=step)


def compute_metrics(
    args: BaseArgs,
    predictions: Prediction,
    actual,
    data_exp_name: str,
    model_name: str,
    step: int,
    pred_s: bool = False,
    save_to_csv: Optional[Path] = None,
    results_csv: str = "",
    use_wandb: bool = False,
) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if args._y_dim == 1:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), TPR(), TNR(), RenyiCorrelation()],
            per_sens_metrics=[ProbPos(), TPR(), TNR()],
        )
        wandb_log(args, metrics, step=step)
    else:
        metrics = run_metrics(
            predictions, actual, metrics=[Accuracy(), RenyiCorrelation()], per_sens_metrics=[]
        )
        wandb_log(args, {f"{data_exp_name} Accuracy": metrics["Accuracy"]}, step=step)
    print(f"Results for {data_exp_name} ({model_name}):")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line

    if save_to_csv is not None and results_csv:
        assert isinstance(save_to_csv, Path)

        full_name = f"{args.dataset}_{data_exp_name}"
        full_name += "_s" if pred_s else "_y"
        if hasattr(args, "eval_on_recon"):
            full_name += "_on_recons" if args.eval_on_recon else "_on_encodings"

        manual_keys = ["seed", "method"]
        manual_values = [str(getattr(args, "seed", args.data_split_seed)), f"\"{model_name}\""]

        results_path = save_to_csv / f"{data_exp_name}_{results_csv}"
        value_list = ",".join(manual_values + [str(v) for v in metrics.values()])
        if not results_path.is_file():
            with results_path.open("w") as f:
                # ========= header =========
                f.write(",".join(manual_keys + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")
        else:
            with results_path.open("a") as f:  # append to existing file
                f.write(value_list + "\n")
        print(f"Results have been written to {results_path.resolve()}")
        if use_wandb:
            for metric_name, value in metrics.items():
                wandb.run.summary[f"{model_name}_{metric_name}"] = value

    return metrics


def fit_classifier(
    args: VaeArgs,
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
    args: VaeArgs,
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
        compute_metrics(
            args,
            preds,
            actual,
            name,
            "classifier",  # not the most descriptive name...
            step=step,
            pred_s=pred_s,
            save_to_csv=save_to_csv,
            results_csv=args.results_csv,
            use_wandb=args.use_wandb,
        )
    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        for eth_clf in [algos.LR(), algos.LRCV(), algos.SVM(kernel="linear")]:
            preds = eth_clf.run(train_data, test_data)

            name = "x_rand_s"
            compute_metrics(
                args,
                preds,
                test_data,
                name,
                eth_clf.name,
                step=step,
                save_to_csv=save_to_csv,
                results_csv=args.results_csv,
                use_wandb=args.use_wandb,
            )


def encode_dataset(
    args: VaeArgs,
    data: Dataset,
    generator: AutoEncoder,
    recons: bool,
    invariant_to: Literal["s", "y"] = "s",
) -> Dataset[Tuple[Tensor, Tensor, Tensor]]:
    print("Encoding dataset...", flush=True)  # flush to avoid conflict with tqdm
    all_x_m = []
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
            if recons:
                zs_m, zy_m = generator.mask(enc, random=True)
                z_m = zs_m if invariant_to == "s" else zy_m
                x_m = generator.decode(z_m, mode="hard")

                if args.dataset in ("celeba", "ssrp", "genfaces"):
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
    print("Done.")

    return encoded_dataset
