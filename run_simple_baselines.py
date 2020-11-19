import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import ethicml as em
import hydra
import numpy as np
import pandas as pd
import torch
from ethicml import implementations
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf.omegaconf import MISSING
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import trange

from fdm.models import Classifier
from fdm.optimisation.train import build_weighted_sampler_from_dataset
from shared.configs import DS, BaseArgs
from shared.data import adult, load_dataset
from shared.models.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import ModelFn, compute_metrics, get_data_dim, random_seed

log = logging.getLogger("BASELINE")

BaselineM = Enum("BaselineM", "cnn dro kamiran")


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


@dataclass
class BaselineArgs:
    # General data set settings
    greyscale: bool = False

    # Optimization settings
    epochs: int = 60
    test_batch_size: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-8
    eta: float = 0.5

    # Misc settings
    method: BaselineM = BaselineM.cnn
    pred_s: bool = False
    oversample: bool = True


@dataclass
class Config(BaseArgs):
    baselines: BaselineArgs = MISSING


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


def run_baseline(cfg: Config) -> None:
    args = cfg.baselines
    if args.method == BaselineM.kamiran:
        if cfg.data.dataset == DS.cmnist:
            raise ValueError(
                "Kamiran & Calders reweighting scheme can only be used with binary sensitive "
                "and target attributes."
            )
        elif cfg.bias.mixing_factor % 1 == 0:
            raise ValueError(
                "Kamiran & Calders reweighting scheme can only be used when there is at least one "
                "sample available for each sensitive/target attribute combination."
            )

    use_gpu = torch.cuda.is_available() and not cfg.misc.gpu < 0
    random_seed(cfg.misc.seed, use_gpu)

    device = torch.device(f"cuda:{cfg.misc.gpu}" if use_gpu else "cpu")
    log.info(f"Running on {device}")

    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)

    train_data = datasets.train
    test_data = datasets.test

    if args.method == BaselineM.kamiran:
        instance_weights = get_instance_weights(train_data, batch_size=args.test_batch_size)
        train_data = IntanceWeightedDataset(train_data, instance_weights=instance_weights)
        train_sampler = None
    else:
        train_sampler = build_weighted_sampler_from_dataset(
            dataset=datasets.train,
            s_count=max(datasets.s_dim, 2),
            test_batch_size=args.test_batch_size,
            batch_size=args.batch_size,
            num_workers=cfg.misc.num_workers,
            oversample=args.oversample,
            balance_hierarchical=False,
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=args.method == BaselineM.kamiran,
        num_workers=cfg.misc.num_workers,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.misc.num_workers,
    )

    input_shape = get_data_dim(train_loader)

    #  Construct the network
    classifier_fn: ModelFn
    if cfg.data.dataset == DS.cmnist:
        classifier_fn = Mp32x23Net(batch_norm=True)
    elif cfg.data.dataset == DS.adult:

        def adult_fc_net(input_dim: int, target_dim: int) -> nn.Sequential:
            encoder = FcNet(hidden_dims=[35])(input_dim=input_dim, target_dim=35)
            classifier = torch.nn.Linear(35, target_dim)
            return nn.Sequential(encoder, classifier)

        classifier_fn = adult_fc_net
    else:
        classifier_fn = Mp64x64Net(batch_norm=True)

    target_dim = datasets.s_dim if args.pred_s else datasets.y_dim

    criterion = None
    if args.method == BaselineM.dro:
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

    if args.method == BaselineM.kamiran:
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

    preds, labels, sens = classifier.predict_dataset(test_data, device=device)
    preds = em.Prediction(pd.Series(preds))
    if cfg.data.dataset == DS.cmnist:
        sens_name = "colour"
    elif cfg.data.dataset == DS.celeba:
        sens_name = cfg.data.celeba_sens_attr.name
    elif cfg.data.dataset == DS.adult:
        sens_name = str(adult.SENS_ATTRS[0])
    else:
        sens_name = "sens_Label"
    sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=[sens_name])
    labels_pd = pd.DataFrame(labels, columns=["labels"])
    actual = em.DataTuple(x=sens_pd, s=sens_pd, y=labels_pd)

    full_name = "baseline"
    if cfg.data.dataset == DS.cmnist:
        full_name += "_greyscale" if args.greyscale else "_color"
    elif cfg.data.dataset == DS.celeba:
        full_name += f"_{str(cfg.data.celeba_sens_attr.name)}"
        full_name += f"_{cfg.data.celeba_target_attr.name}"
    full_name += f"_{str(args.epochs)}epochs.csv"

    compute_metrics(
        cfg=cfg,
        predictions=preds,
        actual=actual,
        exp_name="baseline",
        model_name=args.method.name,
        step=0,
        save_to_csv=Path(to_absolute_path(cfg.misc.save_dir)) if cfg.misc.save_dir else None,
        results_csv=full_name,
        use_wandb=False,
        additional_entries={"eta": args.eta} if args.method == BaselineM.dro else None,
    )


cs = ConfigStore.instance()
cs.store(name="baseline", node=Config)


@hydra.main(config_path="conf", config_name="baseline")
def main(cfg: Config) -> None:
    run_baseline(cfg=cfg)


if __name__ == "__main__":
    main()
