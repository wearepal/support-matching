from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final
from typing_extensions import override

from conduit.data import TernarySample
from conduit.data.datasets import CdtDataLoader, CdtDataset
import conduit.metrics as cdtm
from conduit.models.utils import prefix_keys
from conduit.types import Loss
from loguru import logger
from ranzen.misc import gcopy
from ranzen.torch.loss import CrossEntropyLoss, ReductionType
import torch
from torch import Tensor
from tqdm import tqdm
import wandb

from src.arch.predictors import SetPredictor
from src.data import DataModule, resolve_device
from src.models import OptimizerCfg, SetClassifier, SplitLatentAe
from src.utils import cat, to_item

__all__ = ["NeuralScorer", "NullScorer", "Scorer"]

_PBAR_COL: Final[str] = "#ffe252"


@torch.no_grad()  # pyright: ignore
def _encode_and_score_recons(
    dl: CdtDataLoader[TernarySample],
    *,
    ae: SplitLatentAe,
    device: str | torch.device,
    minimize: bool = False,
) -> tuple[CdtDataset[TernarySample, Tensor, Tensor, Tensor], float]:
    device = resolve_device(device)
    ae.eval()
    ae.to(device)

    zy_ls, y_ls, s_ls = [], [], []
    recon_score = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(
            dl, desc="Encoding dataset and scoring reconstructions", colour=_PBAR_COL
        ):
            batch = batch.to(device, non_blocking=True)
            z = ae.encode(batch.x, transform_zs=False)
            zy_ls.append(z.zy)
            y_ls.append(batch.y)
            s_ls.append(batch.s)
            x_hat = ae.decode(z)
            recon_score -= to_item((batch.x - x_hat).abs().flatten(start_dim=1).mean(-1).sum())
            n += len(batch.x)
    recon_score /= n
    if minimize:
        recon_score *= -1
    zy, y, s = cat(zy_ls, y_ls, s_ls, dim=0, device="cpu")
    logger.info(f"Reconstruction score: {recon_score}")
    return CdtDataset(x=zy, y=y, s=s), recon_score


@torch.no_grad()  # pyright: ignore
def balanced_accuracy(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
    return cdtm.subclass_balanced_accuracy(y_pred=y_pred, y_true=y_true, s=y_true)


@dataclass(eq=False)
class Scorer(ABC):
    @abstractmethod
    def run(
        self,
        dm: DataModule,
        *,
        ae: SplitLatentAe,
        disc: SetPredictor,
        device: torch.device,
        use_wandb: bool = True,
    ) -> float:
        raise NotImplementedError()


@dataclass(eq=False)
class NullScorer(Scorer):
    @override
    def run(
        self,
        dm: DataModule,
        *,
        ae: SplitLatentAe,
        disc: SetPredictor,
        device: torch.device,
        use_wandb: bool = True,
    ) -> float:
        return 0.0


@dataclass(eq=False)
class NeuralScorer(Scorer):
    steps: int = 5_000
    batch_size_tr: int = 16
    batch_size_te: int | None = None
    batch_size_enc: int | None = None

    opt: OptimizerCfg = field(default_factory=OptimizerCfg)
    eval_batches: int = 1000
    inv_score_w: float = 1
    recon_score_w: float = 1
    minimize: bool = False

    @override
    def run(
        self,
        dm: DataModule,
        *,
        ae: SplitLatentAe,
        disc: SetPredictor,
        device: torch.device,
        use_wandb: bool = True,
    ) -> float:
        device = resolve_device(device)
        ae.eval()
        ae.to(device)
        disc.to(device)

        dm = gcopy(dm, batch_size_tr=self.batch_size_tr, deep=False)
        batch_size_enc = self.batch_size_tr if self.batch_size_enc is None else self.batch_size_enc
        logger.info("Encoding training set and scoring its reconstructions")
        dm.train, recon_score_tr = _encode_and_score_recons(
            dl=dm.train_dataloader(eval=True, batch_size=batch_size_enc),
            ae=ae,
            device=device,
            minimize=self.minimize,
        )
        logger.info("Encoding deployment set and scoring its reconstructions")
        dm.deployment, recon_score_dep = _encode_and_score_recons(
            dl=dm.deployment_dataloader(eval=True, batch_size=batch_size_enc),
            ae=ae,
            device=device,
            minimize=self.minimize,
        )
        score = recon_score = self.recon_score_w * 0.5 * (recon_score_tr + recon_score_dep)
        logger.info(f"Aggregate reconstruction score: {recon_score}")

        cross_entropy: Loss = CrossEntropyLoss(reduction=ReductionType.mean)  # type: ignore
        classifier = SetClassifier(model=disc, opt=self.opt, criterion=cross_entropy)
        logger.info("Training invariance-scorer")
        classifier.fit(
            dm.train_dataloader(batch_size=self.batch_size_tr),
            dm.deployment_dataloader(batch_size=self.batch_size_tr),
            steps=self.steps,
            use_wandb=False,
            device=device,
        )
        logger.info("Scoring invariance of encodings")
        batch_size_te = self.batch_size_tr if self.batch_size_te is None else self.batch_size_te
        et = classifier.predict(
            dm.train_dataloader(batch_size=batch_size_te),
            dm.deployment_dataloader(batch_size=batch_size_te),
            device=device,
            max_steps=self.eval_batches,
        )
        inv_score = balanced_accuracy(y_pred=et.y_pred, y_true=et.y_true)
        if not self.minimize:
            inv_score *= -1
        inv_score *= self.inv_score_w
        logger.info(f"Invariance score: {inv_score}")
        score += inv_score
        logger.info(f"Aggregate score: {score}")
        if use_wandb:
            log_dict = {"reconstruction": recon_score, "invariance": inv_score, "total": score}
            wandb.log(prefix_keys(log_dict, prefix="scorer", sep="/"))

        return to_item(score)
