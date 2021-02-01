from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

import ethicml as em
from ethicml import implementations
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig, MISSING
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
from tqdm import trange

from fdm.models import Classifier
from fdm.optimisation.utils import build_weighted_sampler_from_dataset
from shared.configs import (
    AdultConfig,
    BaseConfig,
    CelebaConfig,
    CmnistConfig,
    IsicConfig,
    register_configs,
)
from shared.data import adult, load_dataset
from shared.models.configs import FcNet, Mp32x23Net, Mp64x64Net
from shared.utils import (
    ModelFn,
    as_pretty_dict,
    compute_metrics,
    get_data_dim,
    random_seed,
)
from shared.utils.sampler import StratifiedSampler

LOGGER = logging.getLogger("BASELINE")

BaselineM = Enum("BaselineM", "cnn dro kamiran")


@dataclass
class BaselineArgs:
    _target_: str = "run_baselines.BaselineArgs"

    # General data set settings
    greyscale: bool = False
    labelled_context_set: bool = (
        False  # Whether to train the baseline on the context set with labels
    )

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
class Config(BaseConfig):
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


def run_baseline(cfg: Config) -> None:
    for name, settings in [
        ("bias", cfg.bias),
        ("baselines", cfg.baselines),
        ("data", cfg.data),
        ("misc", cfg.misc),
    ]:
        as_list = sorted(f"{k}: {v}" for k, v in as_pretty_dict(settings).items())
        LOGGER.info(f"{name}: " + "{" + ", ".join(as_list) + "}")
    args = cfg.baselines
    if args.method == BaselineM.kamiran:
        if isinstance(cfg.data, CmnistConfig):
            raise ValueError(
                "Kamiran & Calders reweighting scheme can only be used with binary sensitive "
                "and target attributes."
            )
        elif cfg.bias.mixing_factor % 1 == 0:
            raise ValueError(
                "Kamiran & Calders reweighting scheme can only be used when there is at least one "
                "sample available for each sensitive/target attribute combination."
            )

    use_gpu = torch.cuda.is_available() and not cfg.misc.gpu < 0  # type: ignore
    random_seed(cfg.misc.seed, use_gpu)

    device = torch.device(f"cuda:{cfg.misc.gpu}" if use_gpu else "cpu")
    LOGGER.info(f"Running on {device}")

    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)

    train_data = datasets.train
    if args.labelled_context_set:
        train_data = ConcatDataset([train_data, datasets.context])
    test_data = datasets.test

    train_sampler = build_weighted_sampler_from_dataset(
        dataset=datasets.train,  # type: ignore
        s_count=max(datasets.s_dim, 2),
        batch_size=args.batch_size,
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
    if isinstance(cfg.data, CmnistConfig):
        classifier_fn = Mp32x23Net(batch_norm=True)
    elif isinstance(cfg.data, IsicConfig):

        def resnet50_ft(input_dim: int, target_dim: int) -> ResNet:
            classifier = resnet50(pretrained=True)
            classifier.fc = nn.Linear(classifier.fc.in_features, target_dim)
            return classifier

        classifier_fn = resnet50_ft
    elif isinstance(cfg.data, AdultConfig):

        def adult_fc_net(input_dim: int, target_dim: int) -> nn.Sequential:
            encoder = FcNet(hidden_dims=[35])(input_dim=input_dim, target_dim=35)
            classifier = nn.Linear(35, target_dim)
            return nn.Sequential(encoder, classifier)

        classifier_fn = adult_fc_net
    elif isinstance(cfg.data, CelebaConfig):
        classifier_fn = Mp64x64Net(batch_norm=True)
    else:
        raise NotImplementedError()

    target_dim = datasets.s_dim if args.pred_s else datasets.y_dim

    criterion = None
    if args.method == BaselineM.dro:
        if target_dim == 1:
            criterion = implementations.dro_modules.DROLoss(nn.BCEWithLogitsLoss, eta=args.eta)
        else:
            criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)

    classifier: Classifier = Classifier(
        classifier_fn(input_shape[0], target_dim),
        num_classes=2 if target_dim == 1 else target_dim,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        criterion=criterion,  # type: ignore
    )
    classifier.to(device)

    classifier.fit(
        train_data=train_loader,
        test_data=test_loader,
        epochs=args.epochs,
        device=device,
        pred_s=False,
    )

    preds, labels, sens = classifier.predict_dataset(test_data, device=device)
    preds = em.Prediction(pd.Series(preds))
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
    actual = em.DataTuple(x=sens_pd, s=sens_pd, y=labels_pd)

    full_name = "baseline"
    if isinstance(cfg.data, CmnistConfig):
        full_name += "_greyscale" if args.greyscale else "_color"
    elif isinstance(cfg.data, CelebaConfig):
        full_name += f"_{str(cfg.data.celeba_sens_attr.name)}"
        full_name += f"_{cfg.data.celeba_target_attr.name}"
    elif isinstance(cfg.data, IsicConfig):
        full_name += f"_{str(cfg.data.isic_sens_attr.name)}"
        full_name += f"_{cfg.data.isic_target_attr.name}"
    full_name += f"_{str(args.epochs)}epochs.csv"

    compute_metrics(
        cfg=cfg,
        predictions=preds,
        actual=actual,
        exp_name="baseline",
        model_name=args.method.name,
        step=0,
        s_dim=datasets.s_dim,
        save_to_csv=Path(to_absolute_path(cfg.misc.save_dir)) if cfg.misc.save_dir else None,
        results_csv=full_name,
        use_wandb=False,
        additional_entries={"eta": args.eta} if args.method == BaselineM.dro else None,
    )


cs = ConfigStore.instance()
cs.store(name="baseline_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="baseline")
def main(hydra_config: DictConfig) -> None:
    cfg = Config.from_hydra(hydra_config)
    run_baseline(cfg=cfg)


if __name__ == "__main__":
    main()