from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.arch.autoencoder import ResNetAE, SimpleConvAE
from src.arch.predictors.fcn import Fcn, GatedSetFcn, KvqSetFcn
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    ArtifactLoader,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    NullLabeller,
)
from src.models.discriminator import NeuralDiscriminator
from src.relay import SupMatchRelay


def main() -> None:
    ds_ops = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    ae_arch_ops = [
        Option(SimpleConvAE, name="simple"),
        Option(ResNetAE, name="resnet"),
    ]
    disc_arch_ops = [
        Option(Fcn, name="sw"),
        Option(GatedSetFcn, name="gated"),
        Option(KvqSetFcn, name="kvq"),
    ]
    labeller_ops = [
        Option(ArtifactLoader, name="artifact"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(NullLabeller, name="none"),
    ]

    SupMatchRelay.with_hydra(
        ae_arch=ae_arch_ops,
        disc=[Option(NeuralDiscriminator, name="base")],
        disc_arch=disc_arch_ops,
        ds=ds_ops,
        labeller=labeller_ops,
        instantiate_recursively=False,
        clear_cache=True,
        root="conf",
    )


if __name__ == "__main__":
    main()