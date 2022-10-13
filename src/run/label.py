from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    CentroidalLabelNoiser,
    KmeansOnClipEncodings,
    UniformLabelNoiser,
)
from src.relay import LabelRelay


def main() -> None:
    ds_ops = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    labeller_ops = [
        Option(CentroidalLabelNoiser, name="centroidal_noise"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(UniformLabelNoiser, name="uniform_noise"),
    ]

    LabelRelay.with_hydra(
        ds=ds_ops,
        labeller=labeller_ops,
        instantiate_recursively=False,
        clear_cache=True,
        root="conf",
    )


if __name__ == "__main__":
    main()
