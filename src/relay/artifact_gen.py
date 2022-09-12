from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from conduit.data.datasets.vision import Camelyon17, CelebA
from ranzen.decorators import implements
from ranzen.hydra import Option

from src.data.nih import NIHChestXRayDataset
from src.data.splitter import RandomSplitter
from src.labelling.pipeline import NullLabeller

from .base import BaseRelay

__all__ = ["ArtifactGenRelay"]


@dataclass(eq=False)
class ArtifactGenRelay(BaseRelay):
    @classmethod
    @implements(BaseRelay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            ds=[
                Option(CelebA, name="celeba"),
                Option(Camelyon17, name="camelyon17"),
                Option(NIHChestXRayDataset, name="nih"),
            ],
            split=[Option(RandomSplitter, name="random")],
            labeller=[Option(NullLabeller, name="none")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        assert self.split.save_as_artifact, "splits won't be saved"
        run = self.init_wandb(raw_config, self.labeller)
        self.init_dm()
        run.finish()  # type: ignore
