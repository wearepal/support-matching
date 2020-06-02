from typing import Optional, Literal, List

import torch
from shared.configs import BaseArgs

__all__ = ["ClusterArgs"]


class ClusterArgs(BaseArgs):
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 128
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    seed: int = 42
    eval_on_recon: bool = True

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Training settings
    gpu: int = 0  # which GPU to use (if available)
    resume: Optional[str] = None
    save_dir: str = "experiments/finn"
    evaluate: bool = False
    super_val: bool = False  # Train classifier on encodings as part of validation step.
    super_val_freq: int = 0  # how often to do super val, if 0, do it together with the normal val
    val_freq: int = 5
    log_freq: int = 50
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False
    cluster: Literal["s", "y", "both"] = "both"
    with_supervision: bool = True

    # Encoder settings
    encoder: Literal["ae", "vae", "rotnet"] = "ae"
    enc_levels: int = 4
    enc_channels: int = 64
    init_channels: int = 32
    recon_loss: Literal["l1", "l2", "bce", "huber", "ce", "mixed"] = "l2"
    vgg_weight: float = 0
    std_transform: Literal["softplus", "exp"] = "exp"
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
    pseudo_labeler: Literal["ranking", "cosine"] = "ranking"
    sup_bce_weight: float = 1.0
    k_num: int = 5
    lower_threshold: float = 0.5
    upper_threshold: float = 0.5

    # Classifier
    cl_hidden_dims: List[int] = [256]
    lr: float = 1e-3
    weight_decay: float = 0
    use_multi_head: bool = False

    # Method
    method: Literal["pl_enc", "pl_output", "pl_enc_no_norm", "kmeans"] = "pl_enc"

    _device: torch.device
    _s_dim: int
    _y_dim: int

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = [100, 100]
    labeler_epochs: int = 100
    labeler_wandb: bool = False

    def process_args(self):
        super().process_args()
        if self.super_val_freq < 0:
            raise ValueError("frequency cannot be negative")

    def convert_arg_line_to_args(self, arg_line: str) -> List[str]:
        """Parse each line like a YAML file."""
        if arg_line.startswith(("b_", "c_")):
            arg_line = arg_line[2:]
        return super().convert_arg_line_to_args(arg_line)
