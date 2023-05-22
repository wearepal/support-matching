from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar, Union, overload
from typing_extensions import override

from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import BinarySample, NamedSample, TernarySample
import conduit.metrics as cdtm
from conduit.metrics import hard_prediction
from conduit.models.utils import prefix_keys
from conduit.types import Loss
from loguru import logger
from ranzen.torch import ApproxStratBatchSampler, StratifiedBatchSampler
from ranzen.torch.loss import cross_entropy_loss
from ranzen.torch.utils import inf_generator
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm, trange
import wandb

from src.arch.predictors import SetPredictor
from src.data import EvalTuple
from src.data.utils import resolve_device
from src.evaluation.metrics import EmEvalPair, compute_metrics
from src.utils import cat, soft_prediction, to_item

from .base import Model

__all__ = ["Classifier", "SetClassifier"]


@torch.no_grad()
def cat_cpu_flatten(*ls: list[Tensor], dim: int = 0) -> Iterator[Tensor]:
    for ls_ in ls:
        yield torch.cat(ls_, dim=dim).cpu().flatten()


@dataclass(repr=False, eq=False)
class Classifier(Model):
    """Wrapper for classifier models equipped witht training/inference routines."""

    criterion: Optional[Loss] = None

    @overload
    def predict(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: torch.device,
        with_soft: Literal[False] = ...,
    ) -> EvalTuple[Tensor, None]:
        ...

    @overload
    def predict(
        self, data: CdtDataLoader[TernarySample], *, device: torch.device, with_soft: Literal[True]
    ) -> EvalTuple[Tensor, Tensor]:
        ...

    @torch.no_grad()
    def predict(
        self,
        data: CdtDataLoader[TernarySample],
        *,
        device: Union[torch.device, str],
        with_soft: bool = False,
    ) -> EvalTuple:
        device = resolve_device(device)
        self.to(device)
        self.eval()
        hard_preds_ls, actual_ls, sens_ls, soft_preds_ls = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(data, desc="Generating predictions", colour=self._PBAR_COL):
                batch.to(device)
                logits = self.forward(batch.x)
                hard_preds_ls.append(hard_prediction(logits))
                actual_ls.append(batch.y)
                sens_ls.append(batch.s)
                if with_soft:
                    soft_preds_ls.append(soft_prediction(logits))

        hard_preds, actual, sens = cat_cpu_flatten(hard_preds_ls, actual_ls, sens_ls, dim=0)
        logger.info("Finished generating predictions")

        if with_soft:
            soft_preds = torch.cat(soft_preds_ls, dim=0).cpu()  # don't flatten
            return EvalTuple(y_pred=hard_preds, y_true=actual, s=sens, probs=soft_preds)
        return EvalTuple(y_pred=hard_preds, y_true=actual, s=sens)

    def training_step(self, batch: TernarySample, *, pred_s: bool = False) -> Tensor:
        target = batch.s if pred_s else batch.y
        logits = self.forward(batch.x)
        criterion = cross_entropy_loss if self.criterion is None else self.criterion
        return criterion(input=logits, target=target)

    def fit(
        self,
        train_data: CdtDataLoader[TernarySample],
        *,
        steps: int,
        device: torch.device,
        pred_s: bool = False,
        val_interval: Union[int, float] = 0.1,
        test_data: Optional[CdtDataLoader[TernarySample]] = None,
        grad_scaler: Optional[GradScaler] = None,
        use_wandb: bool = False,
    ) -> None:
        use_amp = grad_scaler is not None
        # Test after every 20% of the total number of training iterations by default.
        if isinstance(val_interval, float):
            val_interval = max(1, round(val_interval * steps))
        self.to(device)
        self.train()

        pbar = trange(steps, desc="Training classifier", colour=self._PBAR_COL)
        train_iter = inf_generator(train_data)
        for step in range(steps):
            batch = next(train_iter)
            batch = batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                loss = self.training_step(batch=batch)
                log_dict = {"train/loss": to_item(loss)}

            if use_amp:  # Apply scaling for mixed-precision training
                loss = grad_scaler.scale(loss)
            loss.backward()  # type: ignore
            self.step(grad_scaler=grad_scaler, scaler_update=True)
            self.optimizer.zero_grad()

            if (test_data is not None) and (step > 0) and (step % val_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    preds_ls, targets_ls, groups_ls = [], [], []
                    for batch in tqdm(
                        test_data, desc="Validating classifier", colour=self._PBAR_COL
                    ):
                        batch = batch.to(device)
                        target = batch.s if pred_s else batch.y
                        with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                            logits = self.forward(batch.x)
                        preds_ls.append(hard_prediction(logits))
                        targets_ls.append(target)
                        groups_ls.append(batch.s)
                preds, targets, groups = cat_cpu_flatten(preds_ls, targets_ls, groups_ls, dim=0)
                pair = EmEvalPair.from_tensors(
                    y_pred=preds, y_true=targets, s=groups, pred_s=pred_s
                )
                metrics = compute_metrics(pair=pair, use_wandb=False, prefix="test", verbose=False)
                log_dict.update(metrics)
                self.model.train()
            if use_wandb:
                wandb.log(log_dict)
            pbar.set_postfix(**log_dict)
            pbar.update()

        pbar.close()
        logger.info("Finished training")


@dataclass(eq=False)
class _ScSample(BinarySample[Tensor]):
    b: int


S = TypeVar("S", bound=NamedSample[Tensor])


@dataclass(repr=False, eq=False)
class SetClassifier(Model):
    """Wrapper for set classifier models equipped witht training/inference routines."""

    model: SetPredictor  # overriding the definition in `Model`
    criterion: Optional[Loss] = None

    @torch.no_grad()
    def _fetch_train_data(
        self, *args: tuple[Iterator[S], int], device: torch.device
    ) -> Iterator[_ScSample]:
        for i, (dl_iter, bs) in enumerate(args):
            batch = next(dl_iter)
            y = torch.full(size=(batch.x.shape[0] // bs,), fill_value=i, dtype=torch.long)
            yield _ScSample(x=batch.x, y=y, b=bs).to(device=device, non_blocking=True)

    def training_step(self, *batches: _ScSample) -> tuple[Tensor, Tensor]:
        logits_ls, target_ls = [], []
        for batch in batches:
            logits_ls.append(self.forward(batch.x, batch_size=batch.b))
            target_ls.append(batch.y)
        logits, target = cat(logits_ls, target_ls)
        criterion = cross_entropy_loss if self.criterion is None else self.criterion
        loss = criterion(input=logits, target=target)
        accuracy = cdtm.accuracy(y_pred=logits, y_true=target)
        return loss, accuracy

    def fit(
        self,
        *dls: CdtDataLoader[S],
        steps: int,
        device: torch.device,
        grad_scaler: Optional[GradScaler] = None,
        use_wandb: bool = False,
    ) -> None:
        use_amp = grad_scaler is not None
        self.to(device)
        self.train()

        iter_bs_pairs = []
        for dl in dls:
            assert isinstance(dl.batch_sampler, (StratifiedBatchSampler, ApproxStratBatchSampler))
            dl_iter = inf_generator(dl)
            iter_bs_pairs.append((dl_iter, dl.batch_sampler.batch_size))

        pbar = trange(steps, desc="Training classifier", colour=self._PBAR_COL)
        for _ in range(steps):
            batches = self._fetch_train_data(*iter_bs_pairs, device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
                loss, accuracy = self.training_step(*batches)
                log_dict = {"loss": to_item(loss), "accuracy": to_item(accuracy)}
                if use_wandb:
                    wandb.log(prefix_keys(log_dict, prefix="test", sep="/"))

            if use_amp:  # Apply scaling for mixed-precision training
                loss = grad_scaler.scale(loss)
            loss.backward()  # type: ignore
            self.step(grad_scaler=grad_scaler, scaler_update=True)
            self.optimizer.zero_grad()
            pbar.set_postfix(**log_dict)
            pbar.update()

        pbar.close()
        logger.info("Finished training")

    @torch.no_grad()
    def predict(
        self, *dls: CdtDataLoader[S], device: Union[torch.device, str], max_steps: int
    ) -> EvalTuple[None, None]:
        device = resolve_device(device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            y_pred_ls = []
            y_true_ls = []
            for i, dl in enumerate(dls):
                assert isinstance(dl.batch_sampler, StratifiedBatchSampler)
                bs = dl.batch_sampler.batch_size
                self.model.batch_size = bs
                pbar = trange(max_steps, desc="Generating predictions", colour=self._PBAR_COL)
                dl_iter = inf_generator(dl)
                for _ in range(max_steps):
                    x = next(dl_iter).x.to(device, non_blocking=True)
                    logits = self.forward(x)
                    y_pred_ls.append(hard_prediction(logits))
                    y_true_ls.append(
                        torch.full(size=(len(logits),), dtype=torch.long, fill_value=i)
                    )
                    pbar.update()
        y_pred, y_true = cat_cpu_flatten(y_pred_ls, y_true_ls, dim=0)
        return EvalTuple(y_pred=y_pred, y_true=y_true)

    @override
    def forward(self, inputs: Tensor, batch_size: Optional[int] = None) -> Tensor:
        return self.model(inputs, batch_size=batch_size)
