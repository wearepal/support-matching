from enum import Enum, auto

__all__ = [
    "AS",
    "BWLoss",
    "CA",
    "CL",
    "DM",
    "DS",
    "Enc",
    "GA",
    "InnRM",
    "InnSc",
    "MMDKer",
    "Meth",
    "PL",
    "QL",
    "RL",
    "VaeStd",
]


class DS(Enum):
    """Which dataset to use."""

    adult = auto()
    cmnist = auto()
    celeba = auto()


class CL(Enum):
    """Which attribute(s) to cluster on."""

    s = auto()
    y = auto()
    both = auto()


class Enc(Enum):
    """Encoder type."""

    ae = auto()
    vae = auto()
    rotnet = auto()


class RL(Enum):
    """Reconstruction loss."""

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


class PL(Enum):
    """Psuedo-labelling method."""

    ranking = auto()
    cosine = auto()


class Meth(Enum):
    """Clustering method."""

    pl_enc = auto()
    pl_enc_no_norm = auto()
    pl_output = auto()
    kmeans = auto()


class DM(Enum):
    """Discriminator method."""

    nn = auto()
    mmd = auto()


class MMDKer(Enum):
    """Kernel to use for MMD (requires discriminator method to be set to MMD)."""

    linear = auto()
    rbf = auto()
    rq = auto()


class BWLoss(Enum):
    """Batch-wise loss."""

    none = auto()
    attention = auto()
    simple = auto()
    transposed = auto()


InnRM = Enum("InnRM", "squeeze haar")  # INN reshape method
InnSc = Enum("InnSc", "none exp sigmoid0_5 add2_sigmoid")  # INN scaling method


class QL(Enum):
    """Quantization level."""

    three = 3
    five = 5
    eight = 8


class GA(Enum):
    """Generated faces attributes."""

    gender = auto()
    age = auto()
    ethnicity = auto()
    eye_color = auto()
    hair_color = auto()
    hair_length = auto()
    emotion = auto()


class CA(Enum):
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


class AS(Enum):
    """Adult dataset splits."""

    Sex = auto()
    Race = auto()
    Race_Binary = auto()
    Race_Sex = auto()
    Custom = auto()
    Nationality = auto()
    Education = auto()
