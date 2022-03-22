"""Simply call the main function."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import attr
from conduit.data.datasets.vision import Camelyon17, CelebA, ColoredMNIST
from omegaconf.dictconfig import DictConfig
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

from shared.configs import Config
from shared.configs.arguments import (
    ASMConfig,
    ClusterConfig,
    DataModuleConfig,
    EncoderConfig,
    LoggingConfig,
    MiscConfig,
    SplitConfig,
)


@attr.define
class ASMRelay(Relay):

    ds: DictConfig
    dm: DictConfig
    split: DictConfig
    misc: DictConfig
    logging: DictConfig
    clust: DictConfig
    enc: DictConfig
    alg: DictConfig
    misc: DictConfig

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
        ds: list[type[Any] | Option],
    ) -> None:
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            enc=[Option(EncoderConfig, "enc")],
            alg=[Option(ASMConfig, "alg")],
            clust=[Option(ClusterConfig, "clust")],
            split=[Option(SplitConfig, "split")],
            logging=[Option(LoggingConfig, "logging")],
            dm=[Option(DataModuleConfig, "dm")],
            misc=[Option(MiscConfig, "misc")],
            ds=ds,
        )

    @implements(Relay)
    def run(self, raw_config: dict[str, Any] | None = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        breakpoint()


if __name__ == "__main__":
    ds_ops: list[type[Any] | Option] = [
        Option(ColoredMNIST, name="cmnist"),  # type: ignore
        Option(CelebA, name="celeba"),  # type: ignore
        Option(Camelyon17, name="camelyon17"),  # type: ignore
    ]

    ASMRelay.with_hydra(
        root="conf2",
        clear_cache=True,
        instantiate_recursively=False,
        ds=ds_ops,
    )
