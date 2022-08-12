from dataclasses import dataclass, field
import logging
import shlex
from typing import Any, Dict, List, Optional

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, MISSING, OmegaConf
import torch

from .enums import (
    AggregatorType,
    ClusteringLabel,
    ClusteringMethod,
    DiscriminatorLoss,
    DiscriminatorMethod,
    EncoderType,
    EvalTrainData,
    MMDKernel,
    PlMethod,
    VaeStd,
    WandbMode,
)

__all__ = [
    "ASMConf",
    "BaseConfig",
    "SplitConf",
    "ClusterConf",
    "Config",
    "MiscConf",
]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


@dataclass
class SplitConf:
    log_dataset: str = ""
    seed: int = 42
    transductive: bool = False  # whether to include the test data in the pool of unlabelled data
    dep_prop: float = 0.4
    test_prop: float = 0.2
    # The propotion of the dataset to use overall (pre-splitting)
    data_prop: Optional[float] = None

    # Dataset manipulation
    dep_subsampling_props: Optional[Dict[int, Any]] = None
    train_subsampling_props: Optional[Dict[int, Any]] = None

    # transforms for image datasets
    train_transforms: Any = None
    test_transforms: Any = None
    dep_transforms: Any = None


@dataclass
class DataModuleConf:
    batch_size_tr: int = 1
    batch_size_te: Optional[int] = None
    num_samples_per_group_per_bag: int = 1
    num_workers: int = 0
    persist_workers: bool = False
    pin_memory: bool = True
    gt_deployment: bool = True
    # Amount of noise to apply to the labels used for balanced sampling
    # -- only applicable when ``gt_deployment=True``
    label_noise: float = 0.0
    seed: int = 47


@dataclass
class LoggingConf:
    exp_group: str = ""  # experiment group; should be unique for a specific setting
    log_method: str = ""  # arbitrary string that's appended to the experiment group name
    mode: WandbMode = WandbMode.online
    save_dir: str = "outputs/asm"
    results_csv: str = ""  # name of CSV file to save results to
    umap: bool = False  # whether to create UMAP plots


@dataclass
class MiscConf:
    # Cluster settings
    cluster_label_file: str = ""
    # General settings
    resume: Optional[str] = None
    evaluate: bool = False
    seed: int = 42
    use_amp: bool = False  # Whether to use mixed-precision training
    gpu: int = 0  # which GPU to use (if available)


@dataclass
class ClusterConf:
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 256
    test_batch_size: Optional[int] = 256
    num_workers: int = 4

    # Evaluation settings
    eval_steps: int = 1000
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Training settings
    val_freq: int = 5
    log_freq: int = 50
    feat_attr: bool = False
    cluster: ClusteringLabel = ClusteringLabel.both
    num_clusters: Optional[int] = None  # this only has an effect if `cluster` is set to `manual`
    with_supervision: bool = True

    # Encoder settings
    encoder: EncoderType = EncoderType.ae
    vgg_weight: float = 0
    vae_std_tform: VaeStd = VaeStd.exp
    kl_weight: float = 1
    elbo_weight: float = 1
    stochastic: bool = False
    enc_path: str = ""
    enc_epochs: int = 100
    enc_lr: float = 1e-3
    enc_wd: float = 0
    enc_wandb: bool = False
    finetune_encoder: bool = False
    finetune_lr: float = 1e-6
    finetune_wd: float = 0
    freeze_layers: int = 0

    # PseudoLabeler
    pseudo_labeler: PlMethod = PlMethod.ranking
    sup_ce_weight: float = 1.0
    sup_bce_weight: float = 1.0
    k_num: int = 5
    lower_threshold: float = 0.5
    upper_threshold: float = 0.5

    # Classifier
    cl_hidden_dims: List[int] = field(default_factory=lambda: [256])
    lr: float = 1e-3
    weight_decay: float = 0
    factorized_s_y: bool = False  # P(s,y) will be factorized to P(s)P(y) with separate outputs

    # Method
    method: ClusteringMethod = ClusteringMethod.pl_enc_no_norm

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = field(default_factory=lambda: [100, 100])
    labeler_epochs: int = 100
    labeler_wandb: bool = False

    def __post_init__(self) -> None:
        if self.cluster is ClusteringLabel.manual and self.num_clusters is None:
            raise ValueError("if 'cluster' is set to 'manual', provide number of clusters")
        if self.cluster is not ClusteringLabel.manual and self.num_clusters is not None:
            raise ValueError("if 'cluster' isn't set to 'manual', don't provide number of clusters")
        if self.cluster is not ClusteringLabel.both and self.factorized_s_y:
            raise ValueError("factorizing s and y requires both y and s")


@dataclass
class ASMConf:
    """Flags for disentangling."""

    # mixup: bool = False
    iters: int = 50_000

    early_stopping: int = 30
    weight_decay: float = 0
    warmup_steps: int = 0
    distinguish_warmup: bool = False
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    recon_detach: bool = True  # Whether to apply the stop gradient operator to the reconstruction.
    eval_on_recon: bool = False

    # Evaluation settings
    eval_steps: int = 10000
    eval_lr: float = 1e-4
    eval_batch_size: int = 256
    balanced_eval: bool = True  # Whether to balance the training set during evaluation
    eval_s_from_zs: Optional[EvalTrainData] = None  # Train a classifier to predict s from zs
    eval_hidden_dims: Optional[List[int]] = None

    # Misc
    validate: bool = True
    val_freq: int = 1_000  # how often to do validation
    log_freq: int = 150
    feat_attr: bool = False

    vgg_weight: float = 0
    vae: bool = False
    vae_std_tform: VaeStd = VaeStd.exp
    stochastic: bool = False

    # Discriminator settings
    adv_loss: DiscriminatorLoss = DiscriminatorLoss.logistic_ns
    # Whether to use the deployment set when computing the encoder's adversarial loss
    double_adv_loss: bool = True
    adv_method: DiscriminatorMethod = DiscriminatorMethod.nn
    mmd_kernel: MMDKernel = MMDKernel.rq
    mmd_scales: List[float] = field(default_factory=list)
    mmd_wts: List[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    adv_hidden_dims: List[int] = field(default_factory=lambda: [256])
    aggregator_type: Optional[AggregatorType] = None
    aggregator_input_dim: int = 32
    aggregator_hidden_dims: List[int] = field(default_factory=list)
    aggregator_kwargs: Dict[str, int] = field(default_factory=dict)

    # Training settings
    lr: float = 1e-3
    adv_lr: float = 3e-4
    enc_loss_w: float = 1
    adv_loss_w: float = 1
    num_adv_updates: int = 3
    distinguish_weight: float = 1
    pred_y_loss_w: float = 1
    pred_s_loss_w: float = 0
    prior_loss_w: float = 0
    pred_y_hidden_dims: Optional[List[int]] = None

    # Encoder settings (that are not shared with the clustering code)
    use_pretrained_enc: bool = False
    zs_dim: int = 1
    s_as_zs: bool = False  # if True, pass `s` instead of `zs` to the decoder for the training set
    s_pred_with_bias: bool = True  # if False, the s predictor has no bias term in the output layer
    ga_steps: int = 1
    max_grad_norm: Optional[float] = 5.0


@dataclass
class BaseConfig:
    """Minimum config needed to do data loading."""

    ds: DictConfig = MISSING
    dm: DataModuleConf = MISSING
    split: SplitConf = MISSING
    misc: MiscConf = MISSING
    logging: LoggingConf = MISSING
    cmd: str = ""  # don't set this in the yaml file (or anywhere really); it will get overwritten


@dataclass
class Config(BaseConfig):
    """Config used for clustering and disentangling."""

    # clust: ClusterConf = MISSING
    enc: DictConfig = MISSING
    alg: ASMConf = MISSING


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program."""
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = internal_config.overrides.task
    return _join([program] + OmegaConf.to_container(args))  # type: ignore[operator]


def _join(split_command: List[str]) -> str:
    """Concatenate the tokens of the list split_command and return a string."""
    return " ".join(shlex.quote(arg) for arg in split_command)
