from __future__ import annotations
from typing import Any, List, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.algs import SupportMatching
from src.algs.adv import Evaluator
from src.arch.autoencoder import ResNetAE, SimpleConvAE
from src.arch.predictors.fcn import Fcn, GatedSetFcn, KvqSetFcn
from src.data import DataModuleConf, DataSplitter
from src.data.nih import NIHChestXRayDataset
from src.labelling.pipeline import (
    ArtifactLoader,
    GroundTruthLabeller,
    KmeansOnClipEncodings,
    NullLabeller,
)
from src.logging import WandbConf
from src.models.autoencoder import SplitLatentAe
from src.models.discriminator import NeuralDiscriminator
from src.relay import SupMatchRelay


def main() -> None:
    ds_ops: List[Union[Type[Any], Option]] = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
        Option(NIHChestXRayDataset, name="nih"),
    ]
    ae_arch_ops: List[Union[Type[Any], Option]] = [
        Option(SimpleConvAE, name="simple"),
        Option(ResNetAE, name="resnet"),
    ]
    disc_arch_ops: List[Union[Type[Any], Option]] = [
        Option(Fcn, name="sw"),
        Option(GatedSetFcn, name="gated"),
        Option(KvqSetFcn, name="kvq"),
    ]
    clust_ops: List[Union[Type[Any], Option]] = [
        Option(ArtifactLoader, name="artifact"),
        Option(GroundTruthLabeller, name="gt"),
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(NullLabeller, name="none"),
    ]

    SupMatchRelay.with_hydra(
        ae=[Option(SplitLatentAe, name="base")],
        ae_arch=ae_arch_ops,
        alg=[Option(SupportMatching, name="base")],
        clust=clust_ops,
        disc=[Option(NeuralDiscriminator, name="base")],
        disc_arch=disc_arch_ops,
        dm=[Option(DataModuleConf, name="base")],
        ds=ds_ops,
        eval=[Option(Evaluator, name="base")],
        split=[Option(DataSplitter, name="base")],
        wandb=[Option(WandbConf, name="base")],
        instantiate_recursively=False,
        clear_cache=True,
        root="conf",
    )


if __name__ == "__main__":
    main()
