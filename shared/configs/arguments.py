from typing import Literal, List, Dict

from typed_flags import TypedFlags

from ethicml.data import GenfacesAttributes

__all__ = ["BaseArgs", "CELEBATTRS"]

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


class BaseArgs(TypedFlags):
    """General data set settings."""

    dataset: Literal["adult", "cmnist", "celeba", "genfaces"] = "cmnist"

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    biased_train: bool = True  # if True, make the training set biased, dependent on mixing factor
    mixing_factor: float = 0.0  # How much of context should be mixed into training?
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    data_split_seed: int = 888
    root: str = ""

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
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
    subsample: Dict[int, float] = {}
    input_noise: bool = True  # add uniform noise to the input
    filter_labels: List[int] = []

    # CelebA settings
    celeba_sens_attr: CELEBATTRS = "Male"
    celeba_target_attr: CELEBATTRS = "Smiling"

    # GenFaces settings
    genfaces_sens_attr: GenfacesAttributes = "gender"
    genfaces_target_attr: GenfacesAttributes = "emotion"

    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    use_wandb: bool = True

    # Global variables
    _s_dim: int
    _y_dim: int

    def process_args(self):
        if not 0 < self.data_pcnt <= 1:
            raise ValueError("data_pcnt has to be between 0 and 1")
