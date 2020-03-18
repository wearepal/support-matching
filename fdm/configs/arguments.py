import argparse

import torch

from typing import List, Optional, Literal
from ethicml.data import GenfacesAttributes

from tap import Tap

__all__ = ["VaeArgs", "SharedArgs", "CELEBATTRS"]

CELEBATTRS = Literal[
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


class SharedArgs(Tap):
    # General data set settings

    dataset: Literal["adult", "cmnist", "celeba", "ssrp", "genfaces"] = "cmnist"

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    task_mixing_factor: float = 0.0  # How much of meta train should be mixed into task train?
    pretrain_pcnt: float = 0.4
    test_pcnt: float = 0.2

    # Adult data set feature settings
    drop_native: bool = True
    drop_discrete: bool = False

    # Colored MNIST settings
    scale: float = 0.02
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    padding: int = 2  # by how many pixels to pad the input images
    quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
    input_noise: bool = True  # add uniform noise to the input

    # CelebA settings
    celeba_sens_attr: CELEBATTRS = "Male"
    celeba_target_attr: CELEBATTRS = "Smiling"

    # GenFaces settings
    genfaces_sens_attr: GenfacesAttributes = "gender"
    genfaces_target_attr: GenfacesAttributes = "emotion"

    # Optimization settings
    early_stopping: int = 30
    epochs: int = 250
    batch_size: int = 128
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    weight_decay: float = 0
    seed: int = 42
    data_split_seed: int = 888
    warmup_steps: int = 0
    gamma: float = 1.0  # Gamma value for Exponential Learning Rate scheduler.
    train_on_recon: bool = False  # whether to train the discriminator on recons or encodings
    recon_detach: bool = False  # Whether to apply the stop gradient operator to the reconstruction.

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
    super_val_freq: int = 0  # how often to do super val, if 0, do it together with the normal val
    val_freq: int = 5
    log_freq: int = 50
    root: str = "data"
    use_wandb: bool = True
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False

    @property
    def _device(self) -> torch.device:
        return self.__device

    @_device.setter
    def _device(self, value: torch.device) -> None:
        self.__device = value

    @property
    def _s_dim(self) -> int:
        return self.__s_dim

    @_s_dim.setter
    def _s_dim(self, value: int) -> None:
        self.__s_dim = value

    @property
    def _y_dim(self) -> int:
        return self.__y_dim

    @_y_dim.setter
    def _y_dim(self, value: int) -> None:
        self.__y_dim = value

    def process_args(self):
        if not 0 < self.data_pcnt <= 1:
            raise ValueError("data_pcnt has to be between 0 and 1")
        if self.super_val_freq < 0:
            raise ValueError("frequency cannot be negative")


class VaeArgs(SharedArgs):
    # VAEsettings
    levels: int = 4
    level_depth: int = 2
    zs_frac: float = 0.33
    zy_frac: float = 0.33
    cond_decoder: bool = True
    init_channels: int = 32
    recon_loss: Literal["l1", "l2", "huber", "ce", "mixed"] = "l2"
    stochastic: bool = True
    vgg_weight: float = 0
    vae: bool = True

    # Discriminator settings
    disc_enc_y_depth: int = 1
    disc_enc_y_channels: int = 256
    disc_enc_s_depth: int = 1
    disc_enc_s_channels: int = 128

    # Training settings
    lr: float = 1e-3
    disc_lr: float = 1e-3
    kl_weight: float = 0.1
    elbo_weight: float = 1
    pred_s_weight: float = 1
    skip_disc_steps: int = 1
