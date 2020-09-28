from typing import Dict, List, Optional

import torch
from typing_extensions import Literal

from shared.configs import BaseArgs, StoreDictKeyPair

__all__ = ["FdmArgs"]


class FdmArgs(BaseArgs):

    # Optimization settings
    early_stopping: int = 30
    iters: int = 50_000
    batch_size: int = 256
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    weight_decay: float = 0
    seed: int = 42
    data_split_seed: int = 888
    warmup_steps: int = 0
    distinguish_warmup: bool = False
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
    recon_detach: bool = True  # Whether to apply the stop gradient operator to the reconstruction.
    eval_on_recon: bool = False

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Misc
    gpu: int = 0  # which GPU to use (if available)
    resume: Optional[str] = None
    save_dir: str = "experiments/finn"
    evaluate: bool = False
    super_val: bool = False  # Train classifier on encodings as part of validation step.
    super_val_freq: int = 1_000  # how often to do super val
    log_freq: int = 50
    use_wandb: bool = True
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False

    _device: torch.device

    # Encoder settings
    use_pretrained_enc: bool = True
    enc_levels: int = 4
    enc_out_dim: int = 64
    enc_init_chan: int = 32
    enc_snorm: bool = False
    zs_frac: float = 0.1

    recon_loss: Optional[Literal["l1", "l2", "bce", "huber", "ce", "mixed"]] = None
    vgg_weight: float = 0
    vae: bool = False
    vae_std_tform: Literal["softplus", "exp"] = "exp"
    stochastic: bool = False

    # INN settings
    use_inn: bool = False
    inn_levels: int = 1
    inn_level_depth: int = 1
    inn_reshape_method: Literal["squeeze", "haar"] = "squeeze"
    inn_coupling_channels: int = 256
    inn_coupling_depth: int = 1
    inn_glow: bool = True
    inn_batch_norm: bool = False
    inn_bn_lag: float = 0  # fraction of current statistics to incorporate into moving average
    inn_factor_splits: Dict[str, int] = {}
    inn_idf: bool = False
    inn_scaling: Literal["none", "exp", "sigmoid0.5", "add2_sigmoid"] = "sigmoid0.5"
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
    disc_method: Literal["nn", "mmd"] = "nn"
    mmd_kernel: Literal["linear", "rbf", "rq"] = "rq"
    mmd_scales: List[float] = None
    mmd_wts: List[float] = None
    mmd_add_dot: float = 0.0

    disc_hidden_dims: List[int] = [256]
    batch_wise_loss: Literal["none", "attention", "simple", "transposed"] = "none"
    batch_wise_latent: int = 32

    # Training settings
    lr: float = 1e-3
    disc_lr: float = 3e-4
    kl_weight: float = 0
    elbo_weight: float = 1
    disc_weight: float = 1
    num_disc_updates: int = 3
    distinguish_weight: float = 1
    pred_weight: float = 1

    # misc
    _cluster_test_acc: float = 0.0
    _cluster_context_acc: float = 0.0

    def add_arguments(self) -> None:
        super().add_arguments()
        self.add_argument(
            "--inn-factor-splits",
            action=StoreDictKeyPair,
            nargs="*",
            default={},
            type=str,
            key_type=str,
            value_type=int,
        )

    def process_args(self) -> None:
        super().process_args()
        if self.recon_loss is None:
            self.recon_loss = "mixed" if self.dataset == "adult" else "l1"
        if self.super_val_freq < 0:
            raise ValueError("frequency cannot be negative")

    def convert_arg_line_to_args(self, arg_line: str) -> List[str]:
        """Parse each line like a YAML file."""
        if arg_line.startswith(("a_", "d_", "e_", "a-", "d-", "e-")):
            arg_line = arg_line[2:]
        return super().convert_arg_line_to_args(arg_line)
