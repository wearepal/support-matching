from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

from ranzen.decorators import implements
from ranzen.hydra import Relay

from advrep.algs.supmatch import SupportMatching
from shared.configs import Config

__all__ = ["ASMRelay"]


@dataclass
class ASMRelay(Relay, Config):
    @implements(Relay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        alg = SupportMatching(cfg=self)
        alg.run()
