from __future__ import annotations
from typing import Any, List, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option

from src.algs import SupportMatching
from src.algs.adv import Evaluator
from src.arch.autoencoder import ResNetAE, SimpleConvAE
from src.arch.predictors.fcn import Fcn, GatedSetFcn, KvqSetFcn
from src.clustering.pipeline import ArtifactLoader, KmeansOnClipEncodings, NoCluster
from src.configs.classes import DataModuleConf
from src.data import DataSplitter
from src.data.nih import NIHChestXRayDataset
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
        Option(KmeansOnClipEncodings, name="kmeans"),
        Option(NoCluster, name="none"),
    ]

    SupMatchRelay.with_hydra(
        alg=[Option(SupportMatching, name="base")],
        ae=[Option(SplitLatentAe, name="base")],
        ae_arch=ae_arch_ops,
        disc=[Option(NeuralDiscriminator, name="base")],
        disc_arch=disc_arch_ops,
        clust=clust_ops,
        dm=[Option(DataModuleConf, name="base")],
        ds=ds_ops,
        eval=[Option(Evaluator, name="base")],
        root="conf",
        split=[Option(DataSplitter, name="base")],
        wandb=[Option(WandbConf, name="base")],
        instantiate_recursively=False,
        clear_cache=True,
    )


if __name__ == "__main__":
    main()
