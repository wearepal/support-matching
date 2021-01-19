from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING
import torch

from .enums import (
    AdultDatasetSplit,
    AggregatorType,
    CelebaAttributes,
    ClusteringLabel,
    ClusteringMethod,
    DicriminatorMethod,
    EncoderType,
    FdmDataset,
    MMDKernel,
    PlMethod,
    QuantizationLevel,
    ReconstructionLoss,
    VaeStd,
)

__all__ = [
    "BaseConfig",
    "BiasConfig",
    "ClusterConfig",
    "Config",
    "DatasetConfig",
    "EncoderConfig",
    "FdmConfig",
    "MiscConfig",
]


@dataclass
class DatasetConfig:
    """General data set settings."""

    _target_: str = "shared.configs.DatasetConfig"

    dataset: FdmDataset = MISSING

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    root: str = ""

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AdultDatasetSplit = AdultDatasetSplit.Sex
    drop_discrete: bool = False
    adult_balanced_test: bool = True
    balance_all_quadrants: bool = True

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
    quant_level: QuantizationLevel = QuantizationLevel.eight  # number of bits that encode color
    input_noise: bool = False  # add uniform noise to the input
    filter_labels: List[int] = field(default_factory=list)
    colors: List[int] = field(default_factory=list)

    # CelebA settings
    celeba_sens_attr: CelebaAttributes = CelebaAttributes.Male
    celeba_target_attr: CelebaAttributes = CelebaAttributes.Smiling


@dataclass
class BiasConfig:

    _target_: str = "shared.configs.BiasConfig"

    # Dataset manipulation
    missing_s: List[int] = MISSING
    mixing_factor: float = 0  # How much of context should be mixed into training?
    adult_biased_train: bool = True  # if True, make the training set biased, based on mixing factor
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[str, float] = field(default_factory=dict)
    subsample_train: Dict[str, float] = field(default_factory=dict)

    log_dataset: str = ""


@dataclass
class MiscConfig:
    _target_: str = "shared.configs.MiscConfig"
    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    exp_group: str = ""  # experiment group; should be unique for a specific setting
    log_method: str = ""  # arbitrary string that's appended to the experiment group name
    use_wandb: bool = True
    gpu: int = 0  # which GPU to use (if available)
    use_amp: bool = True  # Whether to use mixed-precision training
    seed: int = MISSING
    data_split_seed: int = MISSING
    save_dir: str = "experiments/finn"
    results_csv: str = ""  # name of CSV file to save results to
    resume: Optional[str] = None
    evaluate: bool = False
    num_workers: int = 4
    device: str = "cpu"
    use_gpu: bool = False

    def __post_init__(self) -> None:
        self.use_gpu = torch.cuda.is_available() and self.gpu >= 0  # type: ignore
        self.device = f"cuda:{self.gpu}" if self.use_gpu else "cpu"
        if not self.use_gpu:  # If cuda is not enabled, set use_amp to False to avoid warning
            self.use_amp = False


@dataclass
class ClusterConfig:
    """Flags for clustering."""

    _target_: str = "shared.configs.ClusterConfig"

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
    use_multi_head: bool = False

    # Method
    method: ClusteringMethod = ClusteringMethod.pl_enc_no_norm

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = field(default_factory=lambda: [100, 100])
    labeler_epochs: int = 100
    labeler_wandb: bool = False


@dataclass
class EncoderConfig:
    """Flags for the encoder."""

    _target_: str = "shared.configs.EncoderConfig"

    out_dim: int = 64
    levels: int = 4
    init_chans: int = 32
    recon_loss: ReconstructionLoss = ReconstructionLoss.l2


@dataclass
class FdmConfig:
    """Flags for disentangling."""

    _target_: str = "shared.configs.FdmConfig"

    # Optimization settings
    early_stopping: int = 30
    iters: int = 50_000
    batch_size: int = 64
    bag_size: int = 16
    eff_batch_size: int = (
        0  # the total number of samples to be drawn each iteration: bag_size * batch_size
    )
    test_batch_size: Optional[int] = 256
    weight_decay: float = 0
    warmup_steps: int = 0
    distinguish_warmup: bool = False
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
    recon_detach: bool = True  # Whether to apply the stop gradient operator to the reconstruction.
    eval_on_recon: bool = False
    balanced_context: bool = False  # Whether to balance the context set with groundtruth labels
    oversample: bool = False  # Whether to oversample when doing weighted sampling.

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Misc
    validate: bool = True
    val_freq: int = 1_000  # how often to do validation
    log_freq: int = 50
    feat_attr: bool = False

    # Encoder settings
    use_pretrained_enc: bool = True
    snorm: bool = False
    zs_frac: float = 0.1

    vgg_weight: float = 0
    vae: bool = False
    vae_std_tform: VaeStd = VaeStd.exp
    stochastic: bool = False

    num_discs: int = 1
    disc_reset_prob: float = 0.0

    # Discriminator settings
    disc_method: DicriminatorMethod = DicriminatorMethod.nn
    mmd_kernel: MMDKernel = MMDKernel.rq
    mmd_scales: List[float] = field(default_factory=list)
    mmd_wts: List[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    disc_hidden_dims: List[int] = field(default_factory=lambda: [256])
    aggregator_type: AggregatorType = AggregatorType.none
    aggregator_input_dim: int = 32
    aggregator_hidden_dims: List[int] = field(default_factory=list)
    aggregator_kwargs: Dict[str, int] = field(default_factory=dict)

    # Training settings
    lr: float = 1e-3
    disc_lr: float = 3e-4
    kl_weight: float = 0
    elbo_weight: float = 1
    disc_weight: float = 1
    num_disc_updates: int = 3
    distinguish_weight: float = 1
    pred_y_weight: float = 1
    pred_s_weight: float = 0

    def __post_init__(self) -> None:
        self.eff_batch_size = self.batch_size
        if self.aggregator_type != AggregatorType.none:
            self.eff_batch_size *= self.bag_size
        if self.test_batch_size is None:
            self.test_batch_size = self.eff_batch_size


@dataclass
class BaseConfig:
    """Minimum needed config to do data loading."""

    _target_: str = "shared.configs.BaseConfig"

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING
    misc: MiscConfig = MISSING


@dataclass
class Config(BaseConfig):
    """Config used for clustering and disentangling."""

    _target_: str = "shared.configs.Config"

    clust: ClusterConfig = MISSING
    enc: EncoderConfig = MISSING
    fdm: FdmConfig = MISSING
