from enum import Enum, auto

__all__ = [
    "AdultDatasetSplit",
    "AggregatorType",
    "ClusteringLabel",
    "ClusteringMethod",
    "FdmDataset",
    "EncoderType",
    "MMDKernel",
    "PlMethod",
    "QuantizationLevel",
    "ReconstructionLoss",
    "VaeStd",
]


class FdmDataset(Enum):
    """Choice of dataset"""

    adult = auto()
    cmnist = auto()
    celeba = auto()


class ClusteringLabel(Enum):
    """Which attribute(s) to cluster on."""

    s = auto()
    y = auto()
    both = auto()


class EncoderType(Enum):
    """Encoder type."""

    ae = auto()
    vae = auto()
    rotnet = auto()


class ReconstructionLoss(Enum):
    """ Reconstruction loss."""

    l1 = auto()
    l2 = auto()
    bce = auto()
    huber = auto()
    ce = auto()
    mixed = auto()


class VaeStd(Enum):
    """Activation to apply to the VAE's learned std to guarantee non-negativity."""

    softplus = auto()
    exp = auto()


class PlMethod(Enum):
    """Psuedo-labelling method."""

    ranking = auto()
    cosine = auto()


class ClusteringMethod(Enum):
    """Clustering method."""

    pl_enc = auto()
    pl_enc_no_norm = auto()
    pl_output = auto()
    kmeans = auto()


class DicriminatorMethod(Enum):
    """Discriminator method."""

    nn = auto()
    mmd = auto()


class MMDKernel(Enum):
    """Kernel to use for MMD (requires discriminator method to be set to MMD)."""

    linear = auto()
    rbf = auto()
    rq = auto()


class AggregatorType(Enum):
    """Which aggreagation function to use (if any)."""

    none = auto()
    kvq = auto()
    gated = auto()


class QuantizationLevel(Enum):
    """Quantization level."""

    three = 3
    five = 5
    eight = 8


class CelebaAttributes(Enum):
    """CelebA attributes."""

    _5_o_Clock_Shadow = auto()
    Arched_Eyebrows = auto()
    Attractive = auto()
    Bags_Under_Eyes = auto()
    Bald = auto()
    Bangs = auto()
    Big_Lips = auto()
    Big_Nose = auto()
    Black_Hair = auto()
    Blond_Hair = auto()
    Blurry = auto()
    Brown_Hair = auto()
    Bushy_Eyebrows = auto()
    Chubby = auto()
    Double_Chin = auto()
    Eyeglasses = auto()
    Goatee = auto()
    Gray_Hair = auto()
    Heavy_Makeup = auto()
    High_Cheekbones = auto()
    Male = auto()
    Mouth_Slightly_Open = auto()
    Mustache = auto()
    Narrow_Eyes = auto()
    No_Beard = auto()
    Oval_Face = auto()
    Pale_Skin = auto()
    Pointy_Nose = auto()
    Receding_Hairline = auto()
    Rosy_Cheeks = auto()
    Sideburns = auto()
    Smiling = auto()
    Straight_Hair = auto()
    Wavy_Hair = auto()
    Wearing_Earrings = auto()
    Wearing_Hat = auto()
    Wearing_Lipstick = auto()
    Wearing_Necklace = auto()
    Wearing_Necktie = auto()
    Young = auto()


class AdultDatasetSplit(Enum):
    """Adult dataset splits."""

    Sex = auto()
    Race = auto()
    Race_Binary = auto()
    Race_Sex = auto()
    Custom = auto()
    Nationality = auto()
    Education = auto()