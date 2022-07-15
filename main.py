from typing import Any, List, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.hydra import Option
import torch

from advrep.models.autoencoder import ResNetAE, SimpleConvAE
from shared.configs.arguments import (
    ASMConf,
    DataModuleConf,
    LoggingConf,
    MiscConf,
    SplitConf,
)
from shared.data.nih import NIHChestXRayDataset
from shared.relay import ASMRelay

torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
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

    ASMRelay.with_hydra(
        root="conf",
        clear_cache=True,
        instantiate_recursively=False,
        dm=[Option(DataModuleConf, "base")],
        ds=ds_ops,
        split=[Option(SplitConf, "base")],
        enc=ae_ops,
        alg=[Option(ASMConf, "base")],
        logging=[Option(LoggingConf, "base")],
        misc=[Option(MiscConf, "base")],
    )
