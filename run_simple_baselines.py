from abc import ABC, abstractmethod
from fdm.optimisation.train import build_weighted_sampler_from_dataset
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import trange

import ethicml as em
from ethicml import implementations

from fdm.models import Classifier
from shared.configs import BaseArgs
from shared.data import load_dataset
from shared.models.configs.classifiers import mp_32x32_net, fc_net, mp_64x64_net
from shared.utils import random_seed, get_data_dim

BASELINE_METHODS = Literal["cnn", "dro", "kamiran"]


class IntanceWeightedDataset(Dataset):
    def __init__(self, dataset: Dataset, instance_weights: Dataset) -> None:
        super().__init__()
        if len(dataset) != len(instance_weights):
            raise ValueError("Number of instance weights must equal the number of data samples.")
        self.dataset = dataset
        self.instance_weights = instance_weights

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tensor:
        data = self.dataset[index]
        if not isinstance(data, tuple):
            data = (data,)
        iw = self.instance_weights[index]
        if not isinstance(iw, tuple):
            iw = (iw,)
        return data + iw


class BaselineArgs(BaseArgs):
    # General data set settings
    greyscale: bool = False
    seed: int = 0

    # Optimization settings
    epochs: int = 40
    test_batch_size: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 0
    weight_decay = 1e-8
    eta: float = 0.5
    gpu: int = 0
    save_dir: str = "experiments/baseline"

    # Misc settings
    method: BASELINE_METHODS = "cnn"
    pred_s: bool = False

    def process_args(self) -> None:
        if self.method == "kamiran":
            if self.dataset == "cmnist":
                raise ValueError(
                    "Kamiran & Calders reweighting scheme can only be used with binary sensitive and target attributes."
                )
            elif self.mixing_factor % 1 == 0:
                raise ValueError(
                    "Kamiran & Calders reweighting scheme can only be used when there is at least one sample available for each sensitive/target attribute combination."
                )

        return super().process_args()


def get_instance_weights(dataset: Dataset, batch_size: int) -> TensorDataset:
    s_all, y_all = [], []
    for _, s, y in DataLoader(dataset, batch_size=batch_size):
        s_all.append(s.numpy())
        y_all.append(y.numpy())

    s_all = np.concatenate(s_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    s = pd.DataFrame(s_all, columns=["sens"])
    y = pd.DataFrame(y_all, columns=["labels"])
    labels = em.DataTuple(x=y, s=s, y=y)

    instance_weights = em.compute_instance_weights(labels).to_numpy()
    instance_weights = torch.as_tensor(instance_weights).view(-1)
    instance_weights = TensorDataset(instance_weights)

    return instance_weights


class Trainer(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        classifier: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        device: torch.device,
        pred_s: bool = False,
    ) -> None:
        ...


class TrainNaive(Trainer):
    @staticmethod
    def __call__(
        classifier: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        device: torch.device,
        pred_s: bool = False,
    ) -> None:
        pbar = trange(epochs)
        for _ in pbar:
            classifier.train()
            for x, s, y in train_loader:
                target = s if pred_s else y
                x = x.to(device)
                target = target.to(device)

                classifier.zero_grad()
                loss, _ = classifier.routine(x, target)
                loss.backward()
                classifier.step()
        pbar.close()


class TrainKamiran(Trainer):
    @staticmethod
    def __call__(
        classifier: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        device: torch.device,
        pred_s: bool = False,
    ) -> None:
        pbar = trange(epochs)
        for _ in pbar:
            classifier.train()
            for x, s, y, iw in train_loader:
                target = s if pred_s else y
                x = x.to(device)
                target = target.to(device)
                iw = iw.to(device)

                classifier.zero_grad()
                loss, _ = classifier.routine(x, target, instance_weights=iw)
                loss.backward()
                classifier.step()

        pbar.close()


def run_baseline(args: BaselineArgs) -> None:

    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)

    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")
    print(f"Running on {device}")

    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(args)

    train_data = datasets.train
    test_data = datasets.test

    if args.method == "kamiran":
        instance_weights = get_instance_weights(train_data, batch_size=args.test_batch_size)
        train_data = IntanceWeightedDataset(train_data, instance_weights=instance_weights)
        train_sampler = None
    else:
        train_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.train,
            s_dim=datasets.s_dim,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            upsample=args.upsample,
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=args.method == "kamiran",
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers,
    )

    input_shape = get_data_dim(train_loader)

    #  Construct the network
    if args.dataset == "cmnist":
        classifier_fn = mp_32x32_net
    elif args.dataset == "adult":

        def adult_fc_net(in_dim: int, target_dim: int) -> nn.Sequential:
            encoder = fc_net(in_dim, 35, hidden_dims=[35])
            classifier = torch.nn.Linear(35, target_dim)
            return nn.Sequential(encoder, classifier)

        classifier_fn = adult_fc_net
    else:
        classifier_fn = mp_64x64_net

    target_dim = datasets.s_dim if args.pred_s else datasets.y_dim

    criterion = None
    if args.method == "dro":
        if target_dim == 1:
            criterion = implementations.dro_modules.DROLoss(nn.BCEWithLogitsLoss, eta=args.eta)
        else:
            criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)

    # TODO: Using FDM Classifier - should this be clustering classifier?
    classifier: Classifier = Classifier(
        classifier_fn(input_shape[0], target_dim),
        num_classes=2 if target_dim == 1 else target_dim,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        criterion=criterion,
    )
    classifier.to(device)

    if args.method == "kamiran":
        train_fn = TrainKamiran()
    else:
        train_fn = TrainNaive()

    train_fn(
        classifier,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=device,
        pred_s=False,
    )

    preds, ground_truths, sens = classifier.predict_dataset(test_data, device=device)
    preds = em.Prediction(pd.Series(preds))

    ground_truths = em.DataTuple(
        x=pd.DataFrame(sens, columns=["sens"]),
        s=pd.DataFrame(sens, columns=["sens"]),
        y=pd.DataFrame(ground_truths, columns=["labels"]),
    )

    full_name = f"{args.dataset}_{args.method}_baseline"
    if args.dataset == "cmnist":
        full_name += "_greyscale" if args.greyscale else "_color"
    elif args.dataset == "celeba":
        full_name += f"_{str(args.celeba_sens_attr)}"
        full_name += f"_{args.celeba_target_attr}"
    full_name += f"_{str(args.epochs)}epochs.csv"

    metrics = em.run_metrics(
        preds,
        ground_truths,
        metrics=[em.Accuracy(), em.TPR(), em.TNR(), em.RenyiCorrelation()],
        per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR(), em.TNR()],
    )
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()

    if args.method == "dro":
        metrics["eta"] = str(args.eta)

    if args.save_dir is not None:
        save_to_csv = Path(args.save_dir)
        save_to_csv.mkdir(exist_ok=True, parents=True)

        assert isinstance(save_to_csv, Path)
        results_path = save_to_csv / full_name
        value_list = ",".join([str(args.seed)] + [str(v) for v in metrics.values()])
        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                f.write(",".join(["Seed"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")


def main() -> None:
    args = BaselineArgs()
    args.parse_args()
    print(args)
    run_baseline(args=args)


if __name__ == "__main__":
    main()
