from __future__ import annotations
import logging
from pathlib import Path

import ethicml as em
from ethicml import implementations
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet
import wandb
import yaml

from advrep.algs import GDRO, LfF
from advrep.algs.domain_independent import DomainIndependentClassifier
from advrep.models import Classifier
from advrep.optimisation import (
    build_weighted_sampler_from_dataset,
    extract_labels_from_dataset,
)
from shared.configs import register_configs
from shared.configs.arguments import Config
from shared.configs.enums import ContextMode, FsMethod
from shared.data import adult, load_data
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

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class RelabelingDataset(Dataset):
    def __init__(self, dataset: Dataset, s: Tensor, y: Tensor) -> None:
        super().__init__()
        self.dataset = dataset
        self.s = s
        self.y = y

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        return self.dataset[index][0], self.s[index], self.y[index]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


def run(cfg: Config) -> None:
    cfg_dict = {}
    for name, settings in [
        ("bias", cfg.split),
        ("fs_args", cfg.fs_args),
        ("data", cfg.datamodule),
        ("misc", cfg.train),
    ]:
        as_dict = as_pretty_dict(settings)
        cfg_dict[name] = as_dict
        as_list = sorted(f"{k}: {v}" for k, v in as_dict.items())
        LOGGER.info(f"{name}: " + "{" + ", ".join(as_list) + "}")
    cfg_dict = flatten_dict(cfg_dict)
    args = cfg.fs_args
    use_gpu = torch.cuda.is_available() and not cfg.train.gpu < 0  # type: ignore
    random_seed(cfg.train.seed, use_gpu)

    LOGGER.info(
        yaml.dump(as_pretty_dict(cfg), default_flow_style=False, allow_unicode=True, sort_keys=True)
    )
    device = torch.device(f"cuda:{cfg.train.gpu}" if use_gpu else "cpu")
    LOGGER.info(f"Running on {device}")

    # Set up wandb logging
    group = f"{cfg.datamodule.log_name}.{str(args.method.name)}.context_mode_{cfg.fs_args.context_mode.name}"
    if cfg.train.log_method:
        group += "." + cfg.train.log_method
    if cfg.train.exp_group:
        group += "." + cfg.train.exp_group
    if cfg.split.log_dataset:
        group += "." + cfg.split.log_dataset
    local_dir = Path(".", "local_logging")
    local_dir.mkdir(exist_ok=True)
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="suds",
        dir=str(local_dir),
        config=flatten_dict(as_pretty_dict(cfg)),
        group=group if group else None,
        reinit=True,
        mode=cfg.train.wandb.name,
    )
    run.__enter__()  # call the context manager dunders manually to avoid excessive indentation

    #  Load the datasets and wrap with dataloaders
    datasets = load_data(cfg)

    train_data = datasets.train
    test_data = datasets.test
    s_count = max(datasets.dim_s, 2)

    train_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.datamodule.num_workers,
        "shuffle": True,
    }

    input_shape = train_data[0][0].shape
    #  Construct the network
    classifier_fn: ModelFn
    if isinstance(cfg.datamodule, CmnistConfig):
        classifier_fn = Mp32x23Net(batch_norm=True)
    elif isinstance(cfg.datamodule, IsicConfig):

        def resnet50_ft(input_dim: int, target_dim: int) -> ResNet:
            classifier = resnet50(pretrained=True)
            classifier.fc = nn.Linear(classifier.fc.in_features, target_dim)
            return classifier

        classifier_fn = resnet50_ft
    elif isinstance(cfg.datamodule, AdultConfig):

        def adult_fc_net(input_dim: int, target_dim: int) -> nn.Sequential:
            encoder = FcNet(hidden_dims=[35])(input_dim=input_dim, target_dim=35)
            classifier = nn.Linear(35, target_dim)
            return nn.Sequential(encoder, classifier)

        classifier_fn = adult_fc_net
    elif isinstance(cfg.datamodule, CelebaConfig):
        classifier_fn = Mp64x64Net(batch_norm=True)
    else:
        raise NotImplementedError()

    target_dim = datasets.dim_y
    num_classes = max(target_dim, 2)

    classifier_kwargs = {}
    if args.method is FsMethod.lff:
        classifier_cls = LfF
    elif args.method is FsMethod.domind:
        classifier_cls = DomainIndependentClassifier
        classifier_kwargs["num_domains"] = s_count
        target_dim *= s_count
    elif args.method is FsMethod.gdro:
        classifier_cls = GDRO
        s_all, _ = extract_labels_from_dataset(dataset=train_data)
        group_counts = (torch.arange(s_count).unsqueeze(1) == s_all.squeeze()).sum(1).float()
        # process generalization adjustment stuff
        adjustments = args.generalization_adjustment
        if adjustments is not None:
            assert len(adjustments) in (1, s_count)
            if len(adjustments) == 1:
                adjustments = np.array(adjustments * s_count)
            else:
                adjustments = np.array(adjustments)
        classifier_kwargs["group_counts"] = group_counts
        classifier_kwargs["normalize_loss"] = args.normalize_loss
        classifier_kwargs["alpha"] = args.alpha
    else:
        if args.method is FsMethod.dro:
            criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)
        else:
            criterion = "ce"
        classifier_cls = Classifier
        classifier_kwargs["criterion"] = criterion

    classifier = classifier_cls(
        classifier_fn(input_shape[0], num_classes),  # type: ignore
        num_classes=num_classes,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        **classifier_kwargs,
    )
    classifier.to(device)

    cluster_metrics = None
    if args.context_mode is ContextMode.ground_truth:
        LOGGER.info("Using ground-truth labels of context set.")
        train_data = ConcatDataset([train_data, datasets.context])
    # Tehcnicaly this and the method invoked in the subsequent conditional are self-supervised
    # method, however it's far easier to integrate them into this script than to create a new
    # series of models, with the code being as it is at the moment.
    elif args.context_mode is ContextMode.cluster_labels and cfg.train.cluster_label_file:
        LOGGER.info("Using cluster labels as pseudo-labels for context set.")
        cluster_results, cluster_metrics = load_results(cfg)
        subgroup_ids = cluster_results.cluster_ids
        _, s_tr, y_tr = train_data[0]
        y = class_id_to_label(subgroup_ids, s_count=s_count, label="y").view(-1, *s_tr.shape)
        s = class_id_to_label(subgroup_ids, s_count=s_count, label="s").view(-1, *y_tr.shape)
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
            s_count=max(datasets.dim_s, 2),
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
    if isinstance(cfg.datamodule, CmnistConfig):
        sens_name = "colour"
    elif isinstance(cfg.datamodule, CelebaConfig):
        sens_name = cfg.datamodule.celeba_sens_attr.name
    elif isinstance(cfg.datamodule, IsicConfig):
        sens_name = cfg.datamodule.isic_sens_attr.name
    elif isinstance(cfg.datamodule, AdultConfig):
        sens_name = str(adult.SENS_ATTRS[0])
    else:
        sens_name = "sens_Label"
    sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=[sens_name])
    labels_pd = pd.DataFrame(labels, columns=["labels"])
    actual = em.DataTuple(x=sens_pd, s=sens_pd, y=labels_pd)

    full_name = f"baseline_{args.method.name}"
    if isinstance(cfg.datamodule, CmnistConfig):
        full_name += "_greyscale" if args.greyscale else "_color"
    elif isinstance(cfg.datamodule, CelebaConfig):
        full_name += f"_{str(cfg.datamodule.celeba_sens_attr.name)}"
        full_name += f"_{cfg.datamodule.celeba_target_attr.name}"
    elif isinstance(cfg.datamodule, IsicConfig):
        full_name += f"_{str(cfg.datamodule.isic_sens_attr.name)}"
        full_name += f"_{cfg.datamodule.isic_target_attr.name}"
    if args.method is FsMethod.dro:
        full_name += f"_eta_{args.eta}"
    full_name += f"_context_mode={args.context_mode}"
    full_name += f"_epochs={str(args.epochs)}.csv"

    # Compute accuracy + fairness metrics using EthicML
    metrics = compute_metrics(
        predictions=preds,
        actual=actual,
        model_name=args.method.name,
        step=0,
        s_dim=datasets.dim_s,
        use_wandb=True,
        additional_entries=cluster_metrics,
    )
    if args.method == FsMethod.dro:
        metrics.update({"eta": args.eta})
    if cfg.train.save_dir:
        cfg_dict["misc.log_method"] = f"baseline_{args.method.name}"

        results = {}
        results.update(cfg_dict)
        results.update(metrics)
        write_results_to_csv(
            results=results,
            csv_dir=Path(to_absolute_path(cfg.train.save_dir)),
            csv_file=f"{cfg.datamodule.log_name}_{full_name}",
        )
    run.__exit__(None, 0, 0)  # this allows multiple experiments in one python process


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
register_configs()


@hydra.main(config_path="conf", config_name="config")
def main(hydra_config: DictConfig) -> None:
    cfg = Config.from_hydra(hydra_config)
    run(cfg=cfg)


if __name__ == "__main__":
    main()
