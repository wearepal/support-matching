from typing import List, Optional

import torch
from typing_extensions import Literal

from shared.configs import BaseArgs, ParseList

__all__ = ["ClusterArgs"]


class ClusterArgs(BaseArgs):
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 256
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
    val_freq: int = 5
    log_freq: int = 50
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False
    cluster: Literal["s", "y", "both"] = "both"
    with_supervision: bool = True

    # Encoder settings
    encoder: Literal["ae", "vae", "rotnet"] = "ae"
    enc_levels: int = 4
    enc_out_dim: int = 64
    enc_init_chans: int = 32
    recon_loss: Literal["l1", "l2", "bce", "huber", "ce", "mixed"] = "l1"
    vgg_weight: float = 0
    vae_std_tform: Literal["softplus", "exp"] = "exp"
    kl_weight: float = 1
    elbo_weight: float = 1
    stochastic: bool = False
    enc_path: str = ""
    enc_epochs: int = 100
    enc_lr: float = 1e-3
    enc_wd: float = 0
    enc_logging: bool = False
    finetune_encoder: bool = False
    finetune_lr: float = 1e-6
    finetune_wd: float = 0
    freeze_layers: int = 0

    # PseudoLabeler
    pseudo_labeler: Literal["ranking", "cosine"] = "ranking"
    sup_ce_weight: float = 1.0
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
    method: Literal["pl_enc", "pl_output", "pl_enc_no_norm", "kmeans"] = "pl_enc_no_norm"

    _device: torch.device
    _s_dim: int
    _y_dim: int

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = [100, 100]
    labeler_epochs: int = 100
    labeler_logging: bool = False

    def add_arguments(self) -> None:
        super().add_arguments()
        self.add_argument("--cl-hidden-dims", action=ParseList, nargs="*", type=str, value_type=int)
        self.add_argument(
            "--labeler-hidden-dims", action=ParseList, nargs="*", type=str, value_type=int
        )

    def convert_arg_line_to_args(self, arg_line: str) -> List[str]:
        """Parse each line like a YAML file."""
        if arg_line.startswith(("a_", "c_", "e_", "a-", "c-", "e-")):
            arg_line = arg_line[2:]
        return super().convert_arg_line_to_args(arg_line)
