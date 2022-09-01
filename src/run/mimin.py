from __future__ import annotations

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.arch.autoencoder import ResNetAE, SimpleConvAE
from src.arch.predictors.fcn import Fcn
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    ArtifactLoader,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    NullLabeller,
)
from src.models import Model
from src.relay.mimin import MiMinRelay


def main() -> None:
    ds_ops= [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    ae_arch_ops= [
        Option(SimpleConvAE, name="simple"),
        Option(ResNetAE, name="resnet"),
    ]
    disc_arch_ops= [
        Option(Fcn, name="sw"),
    ]
    labeller_ops= [
        Option(ArtifactLoader, name="artifact"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(NullLabeller, name="none"),
    ]

    MiMinRelay.with_hydra(
        ae_arch=ae_arch_ops,
        disc_arch=disc_arch_ops,
        ds=ds_ops,
        labeller=labeller_ops,
        instantiate_recursively=False,
        clear_cache=True,
        root="conf",
    )


if __name__ == "__main__":
    main()
