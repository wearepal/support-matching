from __future__ import annotations
from typing import Any, List, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from advrep.models.autoencoder import ResNetAE, SimpleConvAE
from clustering.artifact import ArtifactLoader
from clustering.pipeline import KmeansOnClipEncodings
from shared.configs.arguments import (
    ASMConf,
    DataModuleConf,
    LoggingConf,
    MiscConf,
    SplitConf,
)
from shared.data.nih import NIHChestXRayDataset
from shared.relay import ASMRelay


def main() -> None:
    ds_ops: List[Union[Type[Any], Option]] = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    ae_ops: List[Union[Type[Any], Option]] = [
        Option(SimpleConvAE, name="simple"),
        Option(ResNetAE, name="resnet"),
    ]
    clust_ops: List[Union[Type[Any], Option]] = [
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(ArtifactLoader, name="artifact"),
    ]

    ASMRelay.with_hydra(
        root="conf",
        clear_cache=True,
        instantiate_recursively=False,
        dm=[Option(DataModuleConf, "base")],
        ds=ds_ops,
        split=[Option(SplitConf, "base")],
        clust=clust_ops,
        enc=ae_ops,
        alg=[Option(ASMConf, "base")],
        logging=[Option(LoggingConf, "base")],
        misc=[Option(MiscConf, "base")],
    )


if __name__ == "__main__":
    main()
