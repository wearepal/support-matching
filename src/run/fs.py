from __future__ import annotations

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.algs.fs import Dro, Erm, Gdro, Jtt, LfF, SdErm
from src.arch.backbones import ResNet, SimpleCNN
from src.arch.predictors import Fcn
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
from src.relay import FsRelay


def main() -> None:
    ds_ops = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    backbone_ops = [
        Option(SimpleCNN, name="simple"),
        Option(ResNet, name="resnet"),
    ]
    pred_ops = [
        Option(Fcn, name="fcn"),
    ]
    labeller_ops = [
        Option(CentroidalLabelNoiser, name="centroidal_noise"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(LabelFromArtifact, name="artifact"),
        Option(NullLabeller, name="none"),
        Option(UniformLabelNoiser, name="uniform_noise"),
    ]
    alg_ops = [
        Option(Dro, name="dro"),
        Option(Erm, name="erm"),
        Option(Gdro, name="gdro"),
        Option(Jtt, name="jtt"),
        Option(LfF, name="lff"),
        Option(SdErm, name="sd"),
    ]

    FsRelay.with_hydra(
        alg=alg_ops,
        backbone=backbone_ops,
        ds=ds_ops,
        labeller=labeller_ops,
        predictor=pred_ops,
        root="conf",
        instantiate_recursively=False,
        clear_cache=True,
    )


if __name__ == "__main__":
    main()
