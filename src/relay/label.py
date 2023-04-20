from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override

from ranzen.hydra import Option

from .base import BaseRelay

__all__ = ["LabelRelay"]


@dataclass(eq=False)
class LabelRelay(BaseRelay):
    @classmethod
    @override
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

    @override
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        run = self.init_wandb(raw_config, self.labeller)
        self.init_dm()
        if run is not None:
            run.finish()
