from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING

from .enums import (
    AS,
    CA,
    CL,
    DM,
    DS,
    GA,
    PL,
    QL,
    RL,
    BWLoss,
    Enc,
    InnRM,
    InnSc,
    Meth,
    MMDKer,
    VaeStd,
)

__all__ = [
    "BaseArgs",
    "BiasConfig",
    "ClusterArgs",
    "Config",
    "DatasetConfig",
    "EncoderConfig",
    "FdmArgs",
    "Misc",
]


@dataclass
class DatasetConfig:
    """General data set settings."""

    dataset: DS = MISSING

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    root: str = ""

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AS = AS.Sex
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
    quant_level: QL = QL.eight  # number of bits that encode color
    input_noise: bool = False  # add uniform noise to the input
    filter_labels: List[int] = field(default_factory=list)
    colors: List[int] = field(default_factory=list)

    # CelebA settings
    celeba_sens_attr: CA = CA.Male
    celeba_target_attr: CA = CA.Smiling

    # GenFaces settings
    genfaces_sens_attr: GA = GA.gender
    genfaces_target_attr: GA = GA.emotion


@dataclass
class BiasConfig:
    # Dataset manipulation
    missing_s: List[int] = MISSING
    mixing_factor: float = MISSING  # How much of context should be mixed into training?
    adult_biased_train: bool = (
        MISSING  # if True, make the training set biased, based on mixing factor
    )
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[str, float] = MISSING
    subsample_train: Dict[str, float] = MISSING

    log_dataset: str = ""


@dataclass
class Misc:
    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    exp_group: str = ""  # experiment group; should be unique for a specific setting
    log_method: str = ""  # arbitrary string that's appended to the experiment group name
    use_wandb: bool = True
    gpu: int = 0  # which GPU to use (if available)
    seed: int = MISSING
    data_split_seed: int = MISSING
    save_dir: str = "experiments/finn"
    results_csv: str = ""  # name of CSV file to save results to
    resume: Optional[str] = None
    evaluate: bool = False
    num_workers: int = 4

    _s_dim: int = MISSING
    _y_dim: int = MISSING
    _device: str = MISSING


@dataclass
class ClusterArgs:
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 256
    test_batch_size: Optional[int] = None

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Training settings
    val_freq: int = 5
    log_freq: int = 50
    feat_attr: bool = False
    cluster: CL = CL.both
    with_supervision: bool = True

    # Encoder settings
    encoder: Enc = Enc.ae
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
    pseudo_labeler: PL = PL.ranking
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
    method: Meth = Meth.pl_enc_no_norm

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = field(default_factory=lambda: [100, 100])
    labeler_epochs: int = 100
    labeler_wandb: bool = False


@dataclass
class EncoderConfig:
    """Flags for the encoder."""

    out_dim: int = 64
    levels: int = 4
    init_chans: int = 32
    recon_loss: RL = RL.l2


@dataclass
class FdmArgs:
    """Flags for disentangling."""

    # Optimization settings
    early_stopping: int = 30
    iters: int = 50_000
    batch_size: int = 256
    test_batch_size: Optional[int] = None
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
    enc_snorm: bool = False
    zs_frac: float = 0.1

    vgg_weight: float = 0
    vae: bool = False
    vae_std_tform: VaeStd = VaeStd.exp
    stochastic: bool = False

    # INN settings
    use_inn: bool = False
    inn_levels: int = 1
    inn_level_depth: int = 1
    inn_reshape_method: InnRM = InnRM.squeeze
    inn_coupling_channels: int = 256
    inn_coupling_depth: int = 1
    inn_glow: bool = True
    inn_batch_norm: bool = False
    inn_bn_lag: float = 0  # fraction of current statistics to incorporate into moving average
    inn_factor_splits: Dict[str, int] = field(default_factory=dict)
    inn_idf: bool = False
    inn_scaling: InnSc = InnSc.sigmoid0_5
    inn_spectral_norm: bool = False
    inn_oxbow_net: bool = False
    inn_lr: float = 3e-4
    nll_weight: float = 1e-2
    recon_stability_weight: float = 0
    path_to_ae: str = ""
    ae_epochs: int = 5
    num_discs: int = 1
    disc_reset_prob: float = 0.0

    # Discriminator settings
    disc_method: DM = DM.nn
    mmd_kernel: MMDKer = MMDKer.rq
    mmd_scales: List[float] = field(default_factory=list)
    mmd_wts: List[float] = field(default_factory=list)
    mmd_add_dot: float = 0.0

    disc_hidden_dims: List[int] = field(default_factory=lambda: [256])
    batch_wise_loss: BWLoss = BWLoss.none
    batch_wise_latent: int = 32
    batch_wise_hidden_dims: List[int] = field(default_factory=list)

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


@dataclass
class BaseArgs:
    """Minimum needed config to do data loading."""

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING
    misc: Misc = MISSING


@dataclass
class Config(BaseArgs):
    """Config used for clustering and disentangling."""

    clust: ClusterArgs = MISSING
    enc: EncoderConfig = MISSING
    fdm: FdmArgs = MISSING
