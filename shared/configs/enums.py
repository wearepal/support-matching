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

DS = Enum("DS", "adult cmnist celeba genfaces")  # dataset
CL = Enum("CL", "s y both")  # cluster target
Enc = Enum("Enc", "ae vae rotnet")  # encoder
RL = Enum("RL", "l1 l2 bce huber ce mixed")  # reconstruction loss
VaeStd = Enum("VaeStd", "softplus exp")  # function used to compute standard deviation in the VAE
PL = Enum("PL", "ranking cosine")  # pseudo labeler
Meth = Enum("Meth", "pl_enc pl_output pl_enc_no_norm kmeans")  # clustering method
InnRM = Enum("InnRM", "squeeze haar")  # INN reshape method
InnSc = Enum("InnSc", "none exp sigmoid0_5 add2_sigmoid")  # INN scaling method
DM = Enum("DM", "nn mmd")  # discriminator method
MMDKer = Enum("MMDKer", "linear rbf rq")  # MMD kernel
BWLoss = Enum("BWLoss", "none attention simple transposed")  # batch-wise loss


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
