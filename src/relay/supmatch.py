from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.decorators import implements
from ranzen.hydra import Option

from src.algs import SupportMatching
from src.algs.adv import Evaluator
from src.arch.autoencoder import AePair
from src.models import SplitLatentAe
from src.models.discriminator import NeuralDiscriminator

from .base import BaseRelay

__all__ = ["SupMatchRelay"]


@dataclass(eq=False)
class SupMatchRelay(BaseRelay):
    ae: DictConfig = MISSING
    ae_arch: DictConfig = MISSING
    disc_arch: DictConfig = MISSING
    disc: DictConfig = MISSING
    eval: DictConfig = MISSING

    @classmethod
    @implements(BaseRelay)
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        ds: List[Option],
        ae_arch: List[Option],
        disc: List[Option],
        disc_arch: List[Option],
        labeller: List[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            alg=[Option(SupportMatching, name="base")],
            ae=[Option(SplitLatentAe, name="base")],
            eval=[Option(Evaluator, name="base")],
            ae_arch=ae_arch,
            ds=ds,
            disc=disc,
            disc_arch=disc_arch,
            labeller=labeller,
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(BaseRelay)
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        run = self.init_wandb(raw_config, self.labeller, self.ae_arch, self.disc_arch)
        dm = self.init_dm()
        alg: SupportMatching = instantiate(self.alg)
        ae_pair: AePair = instantiate(self.ae_arch)(input_shape=dm.dim_x)
        ae: SplitLatentAe = instantiate(self.ae, _partial_=True)(
            model=ae_pair,
            feature_group_slices=dm.feature_group_slices,
        )
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        disc_net, _ = instantiate(self.disc_arch)(
            input_dim=ae.encoding_size.zy,
            target_dim=1,
            batch_size=dm.batch_size_tr,
        )
        disc: NeuralDiscriminator = instantiate(self.disc, _partial_=True)(model=disc_net)
        evaluator: Evaluator = instantiate(self.eval)
        alg.run(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
        run.finish()  # type: ignore
