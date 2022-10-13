from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Option

from .base import BaseRelay

__all__ = ["LabelRelay"]


@dataclass(eq=False)
class LabelRelay(BaseRelay):
    ae: DictConfig = MISSING
    ae_arch: DictConfig = MISSING
    disc_arch: DictConfig = MISSING
    disc: DictConfig = MISSING
    eval: DictConfig = MISSING
    scorer: DictConfig = MISSING
    artifact_name: Optional[str] = None

    @classmethod
    @implements(BaseRelay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        ds: List[Option],
        labeller: List[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            ds=ds,
            labeller=labeller,
        )

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        run = self.init_wandb(raw_config, self.labeller, self.ae_arch, self.disc_arch)
        self.init_dm()
        if run is not None:
            run.finish()
