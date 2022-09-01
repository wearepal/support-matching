from __future__ import annotations
from typing import Any, List, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.algs import Erm, Gdro
from src.arch.backbones import ResNet, SimpleCNN
from src.arch.predictors import Fcn
from src.arch.predictors.fcn import Fcn
from src.data import DataModuleConf, DataSplitter
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    ArtifactLoader,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    NullLabeller,
)
from src.logging import WandbConf
from src.relay import FsRelay


def main() -> None:
    ds_ops: List[Union[Type[Any], Option]] = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    backbone_ops: List[Union[Type[Any], Option]] = [
        Option(SimpleCNN, name="simple"),
        Option(ResNet, name="resnet"),
    ]
    pred_ops: List[Union[Type[Any], Option]] = [
        Option(Fcn, name="fcn"),
    ]
    clust_ops: List[Union[Type[Any], Option]] = [
        Option(ArtifactLoader, name="artifact"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(NullLabeller, name="none"),
    ]
    alg_ops: List[Union[Type[Any], Option]] = [
        Option(Erm, name="erm"),
        Option(Gdro, name="gdro"),
    ]

    FsRelay.with_hydra(
        alg=alg_ops,
        backbone=backbone_ops,
        clear_cache=True,
        clust=clust_ops,
        dm=[Option(DataModuleConf, name="base")],
        ds=ds_ops,
        instantiate_recursively=False,
        predictor=pred_ops,
        root="conf",
        split=[Option(DataSplitter, name="base")],
        wandb=[Option(WandbConf, name="base")],
    )


if __name__ == "__main__":
    main()
