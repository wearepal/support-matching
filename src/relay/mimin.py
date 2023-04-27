from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, MISSING
from ranzen.hydra import Option

from src.algs import MiMin
from src.algs.adv import Evaluator
from src.arch.autoencoder import AePair
from src.models import Model, ModelCfg, SplitLatentAe, SplitLatentAeCfg

from .base import BaseRelay

__all__ = ["MiMinRelay"]


@dataclass(eq=False)
class MiMinRelay(BaseRelay):
    alg: DictConfig = MISSING
    ae_arch: DictConfig = MISSING
    disc_arch: DictConfig = MISSING
    disc: DictConfig = MISSING
    eval: DictConfig = MISSING
    ae: DictConfig = MISSING

    @classmethod
    @override
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        ds: List[Option],
        ae_arch: List[Option],
        disc_arch: List[Option],
        labeller: List[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = False,
    ) -> None:
        configs = dict(
            alg=[Option(MiMin, name="base")],
            ae=[Option(SplitLatentAeCfg, name="base")],
            eval=[Option(Evaluator, name="base")],
            disc=[Option(ModelCfg, name="base")],
            ds=ds,
            ae_arch=ae_arch,
            disc_arch=disc_arch,
            labeller=labeller,
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=instantiate_recursively,
            clear_cache=clear_cache,
            **configs,
        )

    @override
    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        run = self.init_wandb(raw_config, self.labeller, self.ae_arch, self.disc_arch)
        dm = self.init_dm()
        alg: MiMin = instantiate(self.alg)
        ae_pair: AePair = instantiate(self.ae_arch)(input_shape=dm.dim_x)
        ae_conf: SplitLatentAeCfg = instantiate(self.ae)
        ae = SplitLatentAe(model=ae_pair, cfg=ae_conf, feature_group_slices=dm.feature_group_slices)
        logger.info(f"Encoding dim: {ae.latent_dim}, {ae.encoding_size}")
        card_s = dm.card_s
        target_dim = card_s if card_s > 2 else 1
        disc_net, _ = instantiate(self.disc_arch)(
            input_dim=ae.encoding_size.zy,
            target_dim=target_dim,
        )
        disc_conf: ModelCfg = instantiate(self.disc)
        disc = Model(model=disc_net, cfg=disc_conf)
        evaluator: Evaluator = instantiate(self.eval)
        alg.run(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
        if run is not None:
            run.finish()  # type: ignore
