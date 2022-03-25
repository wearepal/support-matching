"""Simply call the main function."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

from advrep.algs.supmatch import SupportMatching
from advrep.models.autoencoder import ResNetAE, SimpleConvAE
from shared.configs import Config
from shared.configs.arguments import (
    ASMConf,
    DataModuleConf,
    LoggingConf,
    MiscConf,
    SplitConf,
)


@dataclass
class ASMRelay(Relay, Config):
    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
        ds: List[Union[Type[Any], Option]],
        enc: List[Union[Type[Any], Option]],
    ) -> None:
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            alg=[Option(ASMConf, "base")],
            split=[Option(SplitConf, "base")],
            logging=[Option(LoggingConf, "base")],
            dm=[Option(DataModuleConf, "base")],
            misc=[Option(MiscConf, "base")],
            enc=enc,
            ds=ds,
        )

    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        alg = SupportMatching(cfg=self)
        alg.run()


if __name__ == "__main__":
    ds_ops: List[Union[Type[Any], Option]] = [
        Option(ColoredMNIST, name="cmnist"),
        Option(CelebA, name="celeba"),
        Option(Camelyon17, name="camelyon17"),
    ]
    ae_ops: List[Union[Type[Any], Option]] = [
        Option(SimpleConvAE, name="simple"),
        Option(ResNetAE, name="resnet"),
    ]

    ASMRelay.with_hydra(
        root="conf",
        clear_cache=True,
        instantiate_recursively=False,
        ds=ds_ops,
        enc=ae_ops,
    )
