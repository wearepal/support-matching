from pathlib import Path
from typing import Optional, Sequence, Tuple

import ethicml as em
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing_extensions import Literal

from fdm.configs import VaeArgs
from fdm.models import AutoEncoder, Classifier
from shared.data import DatasetTriplet, get_data_tuples
from shared.models.configs.classifiers import fc_net, mp_32x32_net, mp_64x64_net
from shared.utils import compute_metrics, make_tuple_from_data, prod

from .utils import log_images


def log_sample_images(args, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(args, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    args: VaeArgs, model, data: DatasetTriplet, step: int, save_to_csv: Optional[Path] = None
) -> None:
    """Compute and log a variety of metrics."""
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
        name="x_zero_s",
        eval_on_recon=args.eval_on_recon,
        pred_s=False,
        save_to_csv=save_to_csv,
    )


def baseline_metrics(args: VaeArgs, data: DatasetTriplet, save_to_csv: Optional[Path]) -> None:
    if args.dataset not in ("cmnist", "celeba", "ssrp", "genfaces"):
        print("Baselines...")
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
                args=args,
                predictions=preds,
                actual=test_data,
                exp_name="original_data",
                model_name=clf.name,
                step=0,
                save_to_csv=save_to_csv,
                results_csv=args.results_csv,
                use_wandb=False,
            )


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


def evaluate(
    args: VaeArgs,
    step: int,
    train_data: "Dataset[Tuple[Tensor, Tensor, Tensor]]",
    test_data: "Dataset[Tuple[Tensor, Tensor, Tensor]]",
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

        preds, labels, sens = clf.predict_dataset(test_loader, device=args._device)
        preds = em.Prediction(hard=pd.Series(preds))
        if args.dataset == "cmnist":
            sens_name = "colour"
        elif args.dataset == "celeba":
            sens_name = args.celeba_sens_attr
        else:
            sens_name = "sens_Label"
        sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=[sens_name])
        labels_pd = pd.DataFrame(labels, columns=["labels"])
        actual = em.DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels_pd)
        compute_metrics(
            args,
            preds,
            actual,
            name,
            "pytorch_classifier",
            step=step,
            save_to_csv=save_to_csv,
            results_csv=args.results_csv,
            use_wandb=args.use_wandb,
        )
    else:
        if not isinstance(train_data, em.DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        for eth_clf in [em.LR(), em.LRCV()]:  # , em.LRCV(), em.SVM(kernel="linear")]:
            preds = eth_clf.run(train_data, test_data)
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
) -> "TensorDataset":
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
                zs_m, zy_m = generator.mask(enc, random=False)
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
