from dataclasses import dataclass, field
import logging
import shlex
from typing import Dict, List, Optional, Type, TypeVar, Union

from conduit.data.datasets.vision.celeba import CelebAttr
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, MISSING, OmegaConf
import torch

from .enums import (
    AdaptationMethod,
    AdultDatasetSplit,
    AggregatorType,
    ClusteringLabel,
    ClusteringMethod,
    ContextMode,
    DiscriminatorLoss,
    DiscriminatorMethod,
    EncoderType,
    EvalTrainData,
    FsMethod,
    IsicAttrs,
    MMDKernel,
    PlMethod,
    QuantizationLevel,
    ReconstructionLoss,
    VaeStd,
    WandbMode,
    ZsTransform,
)

__all__ = [
    "AdaptConfig",
    "AdultConfig",
    "BaseConfig",
    "BiasConfig",
    "CelebaConfig",
    "ClusterConfig",
    "CmnistConfig",
    "Config",
    "DatasetConfig",
    "EncoderConfig",
    "FsConfig",
    "ImageDatasetConfig",
    "IsicConfig",
    "MiscConfig",
    "register_configs",
]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


@dataclass
class DatasetConfig:
    """General data set settings."""

    log_name: str  # don't rely on this to check which dataset is loaded

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    root: str = ""
    transductive: bool = False  # whether to include the test data in the pool of unlabelled data

    num_workers: int = 4
    data_split_seed: int = 42


@dataclass
class AdultConfig(DatasetConfig):
    """Settings specific to the Adult dataset."""

    log_name: str = "adult"

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AdultDatasetSplit = AdultDatasetSplit.Sex
    drop_discrete: bool = False
    adult_balanced_test: bool = True
    balance_all_quadrants: bool = True


@dataclass
class ImageDatasetConfig(DatasetConfig):
    """Settings specific to image datasets."""

    quant_level: QuantizationLevel = QuantizationLevel.eight  # number of bits that encode color
    input_noise: bool = False  # add uniform noise to the input


@dataclass
class CmnistConfig(ImageDatasetConfig):
    """Settings specific to the cMNIST dataset."""

    log_name: str = "cmnist"

    # Colored MNIST settings
    scale: float = 0.0
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    filter_map_labels: Dict[int, int] = field(default_factory=dict)
    colors: List[int] = field(default_factory=list)


@dataclass
class CelebaConfig(ImageDatasetConfig):
    """Settings specific to the CelebA dataset."""

    log_name: str = "celeba"

    # CelebA settings
    celeba_sens_attr: CelebAttr = CelebAttr.Male
    celeba_target_attr: CelebAttr = CelebAttr.Smiling


@dataclass
class IsicConfig(ImageDatasetConfig):
    """Settings specific to the ISIC dataset."""

    log_name: str = "isic"

    # ISIC settings
    isic_sens_attr: IsicAttrs = IsicAttrs.histo
    isic_target_attr: IsicAttrs = IsicAttrs.malignant


@dataclass
class BiasConfig:
    # Dataset manipulation
    missing_s: List[int] = field(default_factory=list)
    mixing_factor: float = 0  # How much of context should be mixed into training?
    adult_biased_train: bool = True  # if True, make the training set biased, based on mixing factor
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Optional[Dict[int, Union[Dict[int, float], float]]] = field(
        default_factory=dict
    )
    subsample_train: Optional[Dict[int, Union[Dict[int, float], float]]] = field(
        default_factory=dict
    )

    log_dataset: str = ""


@dataclass
class MiscConfig:
    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    exp_group: str = ""  # experiment group; should be unique for a specific setting
    log_method: str = ""  # arbitrary string that's appended to the experiment group name
    wandb: WandbMode = WandbMode.online
    save_dir: str = "experiments/finn"
    results_csv: str = ""  # name of CSV file to save results to
    resume: Optional[str] = None
    evaluate: bool = False
    seed: int = MISSING
    use_gpu: bool = False
    use_amp: bool = False  # Whether to use mixed-precision training
    device: str = "cpu"
    gpu: int = 0  # which GPU to use (if available)
    cache_data: bool = False  # if True, all data is cached in memory after being loaded
    umap: bool = False  # whether to create UMAP plots

    def __post_init__(self) -> None:
        # ==== check GPU ====
        self.use_gpu = torch.cuda.is_available() and self.gpu >= 0
        self.device = f"cuda:{self.gpu}" if self.use_gpu else "cpu"
        LOGGER.info(f"{torch.cuda.device_count()} GPUs available. Using device '{self.device}'")

        if not self.use_gpu:  # If cuda is not enabled, set use_amp to False to avoid warning
            self.use_amp = False


@dataclass
class ClusterConfig:
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 256
    test_batch_size: Optional[int] = 256

    # Evaluation settings
    eval_epochs: int = 40
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
class EncoderConfig:
    """Flags that are shared between "adapt" and "clustering" but which don't concern data."""

    out_dim: int = 64
    levels: int = 4
    init_chans: int = 32
    recon_loss: ReconstructionLoss = ReconstructionLoss.l2
    checkpoint_path: str = ""


@dataclass
class AdaptConfig:
    """Flags for disentangling."""

    method: AdaptationMethod = AdaptationMethod.suds
    mixup: bool = False
    batch_size: int = 256
    test_batch_size: Optional[int] = 256
    iters: int = 50_000
    bag_size: int = 16
    balanced_context: bool = False  # Whether to balance the context set with groundtruth labels
    oversample: bool = False  # Whether to oversample when doing weighted sampling.

    early_stopping: int = 30
    weight_decay: float = 0
    warmup_steps: int = 0
    distinguish_warmup: bool = False
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
    recon_detach: bool = True  # Whether to apply the stop gradient operator to the reconstruction.
    eval_on_recon: bool = False

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    eval_batch_size: int = 256
    encode_batch_size: int = 1000
    balanced_eval: bool = False  # Whether to balance the training set during evaluation
    eval_s_from_zs: Optional[EvalTrainData] = None  # Train a classifier to predict s from zs
    eval_hidden_dims: Optional[List[int]] = None

    # Misc
    validate: bool = True
    val_freq: int = 1_000  # how often to do validation
    log_freq: int = 50
    feat_attr: bool = False

    vgg_weight: float = 0
    vae: bool = False
    vae_std_tform: VaeStd = VaeStd.exp
    stochastic: bool = False

    # adversary ensemble (RIP)
    # num_discs: int = 1
    # disc_reset_prob: float = 0.0

    # Discriminator settings
    adv_loss: DiscriminatorLoss = DiscriminatorLoss.logistic_ns
    double_adv_loss: bool = (
        True  # Whether to use the context set when computing the encoder's adversarial loss
    )
    adv_method: DiscriminatorMethod = DiscriminatorMethod.nn
    mmd_kernel: MMDKernel = MMDKernel.rq
    mmd_scales: List[float] = field(default_factory=list)
    mmd_wts: List[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    adv_hidden_dims: List[int] = field(default_factory=lambda: [256])
    aggregator_type: AggregatorType = AggregatorType.none
    aggregator_input_dim: int = 32
    aggregator_hidden_dims: List[int] = field(default_factory=list)
    aggregator_kwargs: Dict[str, int] = field(default_factory=dict)

    # Training settings
    lr: float = 1e-3
    adv_lr: float = 3e-4
    enc_loss_w: float = 0
    gen_loss_weight: float = 1
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
    zs_transform: ZsTransform = ZsTransform.none
    s_as_zs: bool = False  # if True, pass `s` instead of `zs` to the decoder for the training set
    s_pred_with_bias: bool = True  # if False, the s predictor has no bias term in the output layer


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Minimum config needed to do data loading."""

    data: DatasetConfig
    bias: BiasConfig
    misc: MiscConfig

    cmd: str = ""  # don't set this in the yaml file (or anywhere really); it will get overwritten

    @classmethod
    def from_hydra(cls: Type[T], hydra_config: DictConfig) -> T:
        """Instantiate class based on a hydra config."""
        conf: object = OmegaConf.to_object(hydra_config)  # type: ignore
        assert isinstance(conf, cls), f"The given hydra config did not correspond to class {cls}."

        conf.cmd = reconstruct_cmd()
        return conf


@dataclass
class FsConfig:
    """Arguments for models run via the run_fs script."""

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

    # gDRO-specific arguments
    alpha: float = 1.0
    normalize_loss: bool = False
    gamma: float = 0.1
    generalization_adjustment: Optional[List[float]] = None

    # Misc settings
    method: FsMethod = FsMethod.erm
    oversample: bool = True


@dataclass
class Config(BaseConfig):
    """Config used for clustering and disentangling."""

    clust: ClusterConfig = MISSING
    enc: EncoderConfig = MISSING
    adapt: AdaptConfig = AdaptConfig()
    fs_args: FsConfig = FsConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(node=AdultConfig, name="adult", package="data", group="data/schema")
    cs.store(node=CmnistConfig, name="cmnist", package="data", group="data/schema")
    cs.store(node=CelebaConfig, name="celeba", package="data", group="data/schema")
    cs.store(node=IsicConfig, name="isic", package="data", group="data/schema")


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program."""
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = internal_config.overrides.task
    return _join([program] + OmegaConf.to_container(args))  # type: ignore[operator]


def _join(split_command: List[str]) -> str:
    """Concatenate the tokens of the list split_command and return a string."""
    return " ".join(shlex.quote(arg) for arg in split_command)
