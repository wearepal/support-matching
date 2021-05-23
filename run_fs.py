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
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
import wandb
import yaml

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
from shared.utils.loadsave import load_results
from shared.utils.utils import class_id_to_label
from suds.algs import GDRO, LfF
from suds.algs.domain_independent import DomainIndependentClassifier
from suds.models import Classifier
from suds.optimisation.utils import build_weighted_sampler_from_dataset

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Method(Enum):
    erm = auto()
    dro = auto()
    gdro = auto()
    lff = auto()
    domind = auto()


class ContextMode(Enum):
    ground_truth = auto()
    cluster_labels = auto()
    unlabelled = auto()
    propagate = auto()


@dataclass
class FsArgs:
    _target_: str = "run_fs.FsArgs"

    # General data set settings
    greyscale: bool = False
    context_mode: ContextMode = ContextMode.unlabelled

    # Optimization settings
    epochs: int = 60
    test_batch_size: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    weight_decay: float = 0
    eta: float = 0.5
    c: float = 0.0

    # Misc settings
    method: Method = Method.erm
    oversample: bool = True


@dataclass
class FsConfig(BaseConfig):
    fs_args: FsArgs = MISSING


class RelabelingDataset(Dataset):
    def __init__(self, dataset: Dataset, s: Tensor, y: Tensor) -> None:
        super().__init__()
        self.dataset = dataset
        self.s = s
        self.y = y

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        return self.dataset[index], self.s[index], self.y[index]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


def run(cfg: FsConfig) -> None:
    cfg_dict = {}
    for name, settings in [
        ("bias", cfg.bias),
        ("baseline", cfg.fs_args),
        ("data", cfg.data),
        ("misc", cfg.misc),
    ]:
        as_dict = as_pretty_dict(settings)
        cfg_dict[name] = as_dict
        as_list = sorted(f"{k}: {v}" for k, v in as_dict.items())
        LOGGER.info(f"{name}: " + "{" + ", ".join(as_list) + "}")
    cfg_dict = flatten_dict(cfg_dict)
    args = cfg.fs_args
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
        group = f"{cfg.data.log_name}.{str(args.method.name)}"
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
            project="suds",
            dir=str(local_dir),
            config=flatten_dict(as_pretty_dict(cfg)),
            group=group if group else None,
            reinit=True,
        )
        run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

    #  Load the datasets and wrap with dataloaders
    datasets = load_dataset(cfg)

    train_data = datasets.train
    test_data = datasets.test
    s_count = max(datasets.s_dim, 2)

    train_loader_kwargs = {"pin_memory": True, "num_workers": cfg.data.num_workers, "shuffle": True}

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

    classifier_out_dim = max(target_dim, 2)
    classifier_kwargs = {}
    if args.method is Method.lff:
        classifier_cls = LfF
    elif args.method is Method.domind:
        classifier_cls = DomainIndependentClassifier
        classifier_kwargs["num_domains"] = s_count
        target_dim *= s_count
    elif args.method is Method.gdro:
        classifier_cls = GDRO
        classifier_kwargs["c_param"] = args.c
    else:
        if args.method is Method.dro:
            criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)
        else:
            criterion = "ce"
        classifier_cls = Classifier
        classifier_kwargs["criterion"] = criterion

    classifier = classifier_cls(
        classifier_fn(input_shape[0], num_classes),  # type: ignore
        num_classes=classifier_out_dim,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        **classifier_kwargs,
    )
    classifier.to(device)

    if args.context_mode is ContextMode.ground_truth:
        LOGGER.info("Using ground-truth labels of context set.")
        train_data = ConcatDataset([train_data, datasets.context])
    # Tehcnicaly this and the method invoked in the subsequent conditional are self-supervised
    # method, however it's far easier to integrate them into this script than to create a new
    # series of models, with the code being as it is at the moment.
    elif args.context_mode is ContextMode.cluster_labels and cfg.misc.cluster_label_file:
        LOGGER.info("Using cluster labels as pseudo-labels for context set.")
        cluster_results = load_results(cfg)
        subgroup_ids = cluster_results.class_ids
        y = class_id_to_label(subgroup_ids, s_count=s_count, label="y")
        s = class_id_to_label(subgroup_ids, s_count=s_count, label="s")
        context_data = RelabelingDataset(datasets.context, s=s, y=y)
        train_data = ConcatDataset([train_data, context_data])
    elif args.context_mode is ContextMode.propagate:
        LOGGER.info("Propagating labels from training set to context set.")
        classifier.fit(
            train_data=train_data,  # type: ignore
            test_data=test_data,
            epochs=args.epochs,
            device=device,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            **train_loader_kwargs,
        )
        # Generate predictions with the trained model
        y, _, s = classifier.predict_dataset(datasets.context, device=device)
        context_data = RelabelingDataset(datasets.context, s=s, y=y)
        train_data = ConcatDataset([train_data, context_data])
    else:
        train_sampler = build_weighted_sampler_from_dataset(
            dataset=train_data,  # type: ignore
            s_count=max(datasets.s_dim, 2),
            batch_size=args.batch_size,
            oversample=args.oversample,
            balance_hierarchical=True,
        )

        train_loader_kwargs["shuffle"] = False
        train_loader_kwargs["sampler"] = train_sampler

    classifier.fit(
        train_data=train_data,  # type: ignore
        test_data=test_data,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        **train_loader_kwargs,
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
    if args.method is Method.dro:
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
    if args.method == Method.dro:
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
cs.store(name="baseline_schema", node=FsConfig)
register_configs()


@hydra.main(config_path="conf", config_name="baselines")
def main(hydra_config: DictConfig) -> None:
    cfg = FsConfig.from_hydra(hydra_config)
    run(cfg=cfg)


if __name__ == "__main__":
    main()
