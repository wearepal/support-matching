from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from conduit.data import CdtDataLoader, CdtDataset
from conduit.data.structures import TernarySample
import conduit.metrics as cdtm
from omegaconf import DictConfig
from ranzen.misc import gcopy
from ranzen.torch.loss import CrossEntropyLoss, ReductionType
import torch
from torch import Tensor
from tqdm import tqdm

from src.arch.predictors import SetPredictor
from src.data import DataModule, resolve_device
from src.models import Optimizer, SetClassifier, SplitLatentAe
from src.utils import cat, to_item

__all__ = ["Scorer"]


@torch.no_grad()
def _encode_and_score_recons(
    dl: CdtDataLoader[TernarySample],
    *,
    ae: SplitLatentAe,
    device: Union[str, torch.device],
) -> Tuple[CdtDataset[TernarySample, Tensor, Tensor, Tensor], float]:
    ae.eval()
    device = resolve_device(device)
    zy_ls, y_ls, s_ls = [], [], []
    recon_score = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Encoding dataset and scoring reconstructions"):
            batch = batch.to(device, non_blocking=True)
            z = ae.encode(batch.x, transform_zs=False)
            zy_ls.append(z.zy)
            y_ls.append(batch.y)
            s_ls.append(batch.s)
            x_hat = ae.decode(z)
            recon_score -= to_item(torch.abs(batch.x - x_hat).flatten(start_dim=1).mean(-1).sum())
            n += len(batch.x)
    recon_score /= n
    zy, y, s = cat(zy_ls, y_ls, s_ls, dim=0)
    return CdtDataset(x=zy, y=y, s=s), recon_score


@torch.no_grad()
def balanced_accuracy(y_pred: Tensor, *, y_true: Tensor) -> Tensor:
    return cdtm.subclass_balanced_accuracy(y_pred=y_pred, y_true=y_true, s=y_true)


@dataclass(eq=False)
class Scorer:
    steps: int = 5_000
    batch_size_tr: int = 16
    batch_size_te: Optional[int] = None

    optimizer_cls: Optimizer = Optimizer.ADAM
    lr: float = 1.0e-4
    weight_decay: float = 0
    optimizer_kwargs: Optional[DictConfig] = None
    optimizer: torch.optim.Optimizer = field(init=False)
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    test_batches: int = 1000
    disc_score_w: float = 1

    def run(
        self,
        dm: DataModule[CdtDataset],
        *,
        ae: SplitLatentAe,
        disc: SetPredictor,
        device: torch.device,
    ) -> float:
        ae.eval()
        device = resolve_device(device)
        dm = gcopy(dm, batch_size_tr=self.batch_size_tr, deep=False)
        batch_size_te = self.batch_size_tr if self.batch_size_te is None else self.batch_size_te
        dm.train, recon_score_tr = _encode_and_score_recons(
            dl=dm.train_dataloader(eval=True, batch_size=batch_size_te),
            ae=ae,
            device=device,
        )
        dm.deployment, recon_score_dep = _encode_and_score_recons(
            dl=dm.deployment_dataloader(eval=True, batch_size=batch_size_te),
            ae=ae,
            device=device,
        )
        score = (recon_score_tr + recon_score_dep) / 2

        classifier = SetClassifier(
            model=disc,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            criterion=CrossEntropyLoss(reduction=ReductionType.mean),
        )
        classifier.fit(
            dm.train_dataloader(batch_size=self.batch_size_tr),
            dm.deployment_dataloader(batch_size=self.batch_size_tr),
            steps=self.steps,
            use_wandb=False,
            device=device,
        )
        # Generate predictions with the trained model
        et = classifier.predict(
            dm.train_dataloader(batch_size=batch_size_te),
            dm.deployment_dataloader(batch_size=batch_size_te),
            device=device,
            max_steps=self.test_batches,
        )
        disc_score = 1.0 - balanced_accuracy(y_pred=et.y_pred, y_true=et.y_true)
        score += self.disc_score_w * disc_score

        return to_item(score)
