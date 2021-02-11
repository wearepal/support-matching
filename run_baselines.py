from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
import wandb
import yaml

from fdm.baselines import GDRO, LfF
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
    flatten_dict,
    random_seed,
    write_results_to_csv,
)

LOGGER = logging.getLogger("BASELINE")


class BaselineM(Enum):
    erm = auto()
    dro = auto()
    gdro = auto()
    lff = auto()


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
    c: float = 0.0

    # Misc settings
    method: BaselineM = BaselineM.erm
    oversample: bool = True


@dataclass
class Config(BaseConfig):
    baseline: BaselineArgs = MISSING


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
    cfg_dict = {}
    for name, settings in [
        ("bias", cfg.bias),
        ("baseline", cfg.baseline),
        ("data", cfg.data),
        ("misc", cfg.misc),
    ]:
        as_dict = as_pretty_dict(settings)
        cfg_dict[name] = as_dict
        as_list = sorted(f"{k}: {v}" for k, v in as_dict.items())
        LOGGER.info(f"{name}: " + "{" + ", ".join(as_list) + "}")
    cfg_dict = flatten_dict(cfg_dict)
    args = cfg.baseline
    use_gpu = torch.cuda.is_available() and not cfg.misc.gpu < 0  # type: ignore
    random_seed(cfg.misc.seed, use_gpu)

    LOGGER.info(
        yaml.dump(as_pretty_dict(cfg), default_flow_style=False, allow_unicode=True, sort_keys=True)
    )
    device = torch.device(f"cuda:{cfg.misc.gpu}" if use_gpu else "cpu")
    LOGGER.info(f"Running on {device}")

    # Set up wandb logging
    run = None
    if cfg.misc.use_wandb:
        project_suffix = f"-{cfg.data.log_name}" if not isinstance(cfg.data, CmnistConfig) else ""
        group = ""
        if cfg.misc.log_method:
            group += cfg.misc.log_method
        if cfg.misc.exp_group:
            group += "." + cfg.misc.exp_group
        if cfg.bias.log_dataset:
            group += "." + cfg.bias.log_dataset
        local_dir = Path(".", "local_logging")
        local_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity="predictive-analytics-lab",
            project="fdm-baselines" + project_suffix,
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(cfg)),
            group=group if group else None,
            reinit=True,
        )
        run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)

    train_data = datasets.train
    if args.labelled_context_set:
        train_data = ConcatDataset([train_data, datasets.context])
    test_data = datasets.test

    train_sampler = build_weighted_sampler_from_dataset(
        dataset=train_data,  # type: ignore
        s_count=max(datasets.s_dim, 2),
        batch_size=args.batch_size,
        oversample=args.oversample,
        balance_hierarchical=not args.labelled_context_set,
    )
    train_loader_kwargs = {
        "sampler": train_sampler,
        "pin_memory": True,
        "num_workers": cfg.misc.num_workers,
    }

    input_shape = train_data[0][0].shape

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

    target_dim = datasets.y_dim
    num_classes = max(target_dim, 2)

    classifier_cls: type[Classifier] | type[LfF]
    classifier_kwargs = {}
    if args.method is BaselineM.lff:
        classifier_cls = LfF
    elif args.method is BaselineM.gdro:
        classifier_cls = GDRO
        classifier_kwargs["c_param"] = args.c
    else:
        if args.method is BaselineM.dro:
            criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)
        else:
            criterion = "ce"
        classifier_cls = Classifier
        classifier_kwargs["criterion"] = criterion

    classifier = classifier_cls(
        classifier_fn(input_shape[0], num_classes),  # type: ignore
        num_classes=max(target_dim, 2),
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        **classifier_kwargs,
    )
    classifier.to(device)
    classifier.fit(
        train_data=train_data,  # type: ignore
        test_data=test_data,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        # **train_loader_kwargs,
    )

    # Generate predictions with the trained model
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

    full_name = f"baseline_{args.method.name}"
    if isinstance(cfg.data, CmnistConfig):
        full_name += "_greyscale" if args.greyscale else "_color"
    elif isinstance(cfg.data, CelebaConfig):
        full_name += f"_{str(cfg.data.celeba_sens_attr.name)}"
        full_name += f"_{cfg.data.celeba_target_attr.name}"
    elif isinstance(cfg.data, IsicConfig):
        full_name += f"_{str(cfg.data.isic_sens_attr.name)}"
        full_name += f"_{cfg.data.isic_target_attr.name}"
    if args.method is BaselineM.dro:
        full_name += f"_eta_{args.eta}"
    full_name += f"_{str(args.epochs)}epochs.csv"

    # Compute accuracy + fairness metrics using EthicML
    metrics = compute_metrics(
        cfg=cfg,
        predictions=preds,
        actual=actual,
        model_name=args.method.name,
        step=0,
        s_dim=datasets.s_dim,
        use_wandb=True,
    )
    if args.method == BaselineM.dro:
        metrics.update({"eta": args.eta})
    if cfg.misc.save_dir:
        cfg_dict["misc.log_method"] = f"baseline_{args.method.name}"

        results = {}
        results.update(cfg_dict)
        results.update(metrics)
        write_results_to_csv(
            results=results,
            csv_dir=Path(to_absolute_path(cfg.misc.save_dir)),
            csv_file=f"{cfg.data.log_name}_{full_name}",
        )
    if run is not None:
        run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process


cs = ConfigStore.instance()
cs.store(name="baseline_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="baselines")
def main(hydra_config: DictConfig) -> None:
    cfg = Config.from_hydra(hydra_config)
    run_baseline(cfg=cfg)


if __name__ == "__main__":
    main()
