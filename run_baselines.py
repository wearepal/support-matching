from __future__ import annotations
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet

from fdm.baselines.lff import LfF
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
    get_data_dim,
    random_seed,
    write_results_to_csv,
)

LOGGER = logging.getLogger("BASELINE")

BaselineM = Enum("BaselineM", "cnn dro lff")


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
    cfg_dict = {}
    for name, settings in [
        ("bias", cfg.bias),
        ("baselines", cfg.baselines),
        ("data", cfg.data),
        ("misc", cfg.misc),
    ]:
        as_dict = as_pretty_dict(settings)
        cfg_dict[name] = as_dict
        as_list = sorted(f"{k}: {v}" for k, v in as_dict.items())
        LOGGER.info(f"{name}: " + "{" + ", ".join(as_list) + "}")
    cfg_dict = flatten_dict(cfg_dict)
    args = cfg.baselines
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

    target_dim = datasets.s_dim if args.pred_s else datasets.y_dim
    num_classes = max(target_dim, 2)

    classifier_cls: type[Classifier] | type[LfF]
    if args.method is BaselineM.lff:
        classifier_cls = LfF
        out_dim = num_classes
    else:
        criterion = None
        if args.method is BaselineM.dro:
            if target_dim == 1:
                criterion = implementations.dro_modules.DROLoss(nn.BCEWithLogitsLoss, eta=args.eta)
            else:
                criterion = implementations.dro_modules.DROLoss(nn.CrossEntropyLoss, eta=args.eta)
        classifier_cls = Classifier
        out_dim = target_dim

    classifier = classifier_cls(
        classifier_fn(input_shape[0], out_dim),  # type: ignore
        num_classes=max(target_dim, 2),
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
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

    metrics = compute_metrics(
        cfg=cfg,
        predictions=preds,
        actual=actual,
        model_name=args.method.name,
        step=0,
        s_dim=datasets.s_dim,
        use_wandb=False,
    )
    if args.method == BaselineM.dro:
        metrics.update({"eta": args.eta})
    if cfg.misc.save_dir:
        cfg.misc.log_method = f"baseline_{args.method.name}"

        results = {}
        results.update(cfg_dict)
        results.update(metrics)
        write_results_to_csv(
            results=results,
            csv_dir=Path(to_absolute_path(cfg.misc.save_dir)),
            csv_file=f"{cfg.data.log_name}_{full_name}",
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
