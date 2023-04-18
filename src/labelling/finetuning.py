from itertools import islice
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple, Union

from attrs import define, field
from conduit import metrics as cdtm
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.models.utils import prefix_keys
from loguru import logger
from ranzen import gcopy
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor, nn, optim
import torch.nn as nn
from tqdm import tqdm
import wandb

from src.data import DataModule, labels_to_group_id, resolve_device

__all__ = ["FineTuner"]


@define(eq=False)
class FineTuner(nn.Module):
    _PBAR_COL: ClassVar[str] = "#ffe252"

    batch_size: int = 16
    steps: int = 2000
    val_freq: Union[int, float] = 0.1
    val_batches: Union[int, float] = 1.0
    lr: float = 1e-5
    device: Union[int, str, torch.device] = 0
    save_path: Optional[str] = None
    loss_fn: CrossEntropyLoss = field(default=CrossEntropyLoss)
    _LOG_PREFIX: ClassVar[str] = "fine-tuning"

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        if isinstance(self.val_freq, float) and (not (0 <= self.val_freq <= 1)):
            raise AttributeError("If 'val_freq' is a float, it must be in the range [0, 1].")
        if isinstance(self.val_batches, float) and (not (0 <= self.val_batches <= 1)):
            raise AttributeError("If 'val_batches' is a float, it must be in the range [0, 1].")

    def run(self, dm: DataModule, *, backbone: nn.Module, out_dim: int) -> nn.Sequential:
        dm = gcopy(dm, deep=False)
        dm.cfg.batch_size_tr = self.batch_size
        device = resolve_device(self.device)

        logger.info(f"Initialising predictor for fine-tuning.")
        model = nn.Sequential(
            backbone, nn.Linear(in_features=out_dim, out_features=dm.num_sources_dep)
        )
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)

        logger.info(f"Starting fine-tuning routine.")
        self.train_loop(
            model,
            train_loader=dm.train_dataloader(balance=True),
            val_loader=dm.test_dataloader(),
            optimizer=optimizer,
            steps=self.steps,
            device=device,
            card_s=dm.card_s,
        )
        logger.info("Fine-tuning complete!")
        if self.save_path is not None:
            torch.save(backbone.state_dict(), f=self.save_path)
            logger.info(f"Fine-tuned model saved to '{Path(self.save_path).resolve()}'")
        model.cpu()
        return model

    def train_loop(
        self,
        model: nn.Module,
        *,
        train_loader: CdtDataLoader[TernarySample[Tensor]],
        val_loader: CdtDataLoader[TernarySample[Tensor]],
        optimizer: optim.Optimizer,
        steps: int,
        device: torch.device,
        card_s: int,
    ) -> None:
        model.train()
        pbar = tqdm(
            islice(train_loader, steps),
            total=steps,
            colour=self._PBAR_COL,
        )
        last_acc = None
        val_freq = max(
            (self.val_freq if isinstance(self.val_freq, int) else round(self.val_freq * steps)),
            1,
        )
        logger.info(f"Set to validate every {val_freq} steps.")
        for step, sample in enumerate(pbar, start=1):
            sample.to(device, non_blocking=True)
            group_id = labels_to_group_id(s=sample.s, y=sample.y, s_count=card_s)
            _, loss = self.train_step(model, inputs=sample.x, targets=group_id, optimizer=optimizer)
            to_log = {"loss": loss}
            to_log = prefix_keys(to_log, prefix=self._LOG_PREFIX)
            wandb.log(to_log, step=step)

            if (step % val_freq) == 0:
                last_acc = self.validate(model, val_loader=val_loader, device=device, step=step)
                model.train()
            pbar.set_postfix(loss=loss, last_acc=last_acc)

    def train_step(
        self,
        model: nn.Module,
        *,
        inputs: Tensor,
        targets: Tensor,
        optimizer: optim.Optimizer,
    ) -> Tuple[Tensor, float]:
        output = model(inputs)
        loss = self.loss_fn(input=output, target=targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return output, loss.item()

    @torch.no_grad()
    def predict_loop(
        self,
        model: nn.Module,
        *,
        val_loader: CdtDataLoader[TernarySample[Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        model.eval()
        all_preds: List[Tensor] = []
        all_s: List[Tensor] = []
        all_y: List[Tensor] = []
        with torch.no_grad():
            val_batches = (
                self.val_batches
                if isinstance(self.val_batches, int)
                else round(self.val_batches * len(val_loader))
            )
            for sample in tqdm(
                islice(val_loader, val_batches),
                total=val_batches,
                desc="Generating predictions",
                colour=self._PBAR_COL,
            ):
                logits = model(sample.x.to(device, non_blocking=True))
                all_preds.append(torch.argmax(logits, dim=-1).detach().cpu())
                all_s.append(sample.s)
                all_y.append(sample.y)

        preds = torch.cat(all_preds)
        s = torch.cat(all_s)
        y = torch.cat(all_y)
        return preds, s, y

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        *,
        val_loader: CdtDataLoader[TernarySample[Tensor]],
        device: torch.device,
        step: int,
    ) -> float:
        preds, s, y = self.predict_loop(model, val_loader=val_loader, device=device)
        group_ids = labels_to_group_id(s=s, y=y, s_count=2)
        acc = cdtm.accuracy(y_pred=preds, y_true=group_ids).item()
        rob_acc = cdtm.robust_accuracy(y_pred=preds, y_true=group_ids, s=group_ids).item()
        to_log = {
            f"Accuracy": acc,
            f"Robust_Accuracy": rob_acc,
        }
        to_log = prefix_keys(to_log, prefix=self._LOG_PREFIX)
        wandb.log(to_log, step=step)
        return acc
