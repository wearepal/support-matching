from __future__ import annotations

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.arch.autoencoder import AeFromArtifact, ResNetAE, SimpleConvAE, VqGanAe
from src.arch.predictors.fcn import Fcn
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    LabelFromArtifact,
    NullLabeller,
    UniformLabelNoiser,
)
from src.relay.mimin import MiMinRelay


def main() -> None:
    ds_ops = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    ae_arch_ops = [
        Option(AeFromArtifact, name="artifact"),
        Option(ResNetAE, name="resnet"),
        Option(SimpleConvAE, name="simple"),
        Option(VqGanAe, name="vqgan"),
    ]
    disc_arch_ops = [Option(Fcn, name="fcn")]
    labeller_ops = [
        Option(CentroidalLabelNoiser, name="centroidal_noise"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(LabelFromArtifact, name="artifact"),
        Option(NullLabeller, name="none"),
        Option(UniformLabelNoiser, name="uniform_noise"),
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
