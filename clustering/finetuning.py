from dataclasses import dataclass, field
from itertools import islice
from typing import List, Tuple

import clip
from conduit import metrics as cdtm
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.types import Loss
from ranzen import gcopy
from ranzen.torch import CrossEntropyLoss, DcModule
import torch
from torch import Tensor, nn, optim
import torch.nn as nn
from tqdm import tqdm
import wandb

from shared.data import DataModule
from shared.data.utils import labels_to_group_id

__all__ = ["FineTuner"]


@dataclass(eq=False)
class FineTuner(DcModule):

    batch_size: int = 10
    steps: int = 2000
    val_freq: int = 100
    val_batches: int = 400
    lr: float = 1e-5
    gpu: int = 0
    save_path: str = ""
    download_root: str = ""
    model_path: str = "./finetuned.pt"
    loss_fn: CrossEntropyLoss = field(init=False, default_factory=CrossEntropyLoss)

    def run(self, dm: DataModule, *, backbone: nn.Module, out_dim: int) -> None:
        dm = gcopy(dm)
        dm.batch_size_tr = self.batch_size
        use_gpu = torch.cuda.is_available() and self.gpu >= 0
        device = torch.device(f"cuda:{self.gpu}" if use_gpu else "cpu")

        model = nn.Sequential(
            backbone, nn.Linear(in_features=out_dim, out_features=dm.num_sources_dep)
        )
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)

        self.train_loop(
            model,
            train_loader=dm.train_dataloader(balance=True),
            val_loader=dm.test_dataloader(),
            optimizer=optimizer,
            steps=self.steps,
            device=device,
            card_s=dm.card_s,
        )
        torch.save(backbone.state_dict(), f=self.model_path)

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
        pbar = tqdm(islice(train_loader, steps), total=steps)
        last_acc = 0.0
        for step, sample in enumerate(pbar, start=1):
            x = sample.x.to(device, non_blocking=True)
            s = sample.s.to(device, non_blocking=True)
            y = sample.y.to(device, non_blocking=True)
            group_id = labels_to_group_id(s=s, y=y, s_count=card_s)

            _, loss = self.train_step(
                model, inputs=x, targets=group_id, loss_fn=loss_fn, optimizer=optimizer
            )

            to_log = {
                "loss": loss,
                "s1_share": sample.s.float().mean().item(),
                "y1_share": sample.y.float().mean().item(),
            }
            wandb.log(to_log, step=step)
            if step % self.val_freq == 0:
                last_acc = self.validate(model, val_loader=val_loader, device=device, step=step)
            pbar.set_postfix(loss=loss, last_acc=last_acc)

    def train_step(
        self,
        model: nn.Module,
        *,
        inputs: Tensor,
        targets: Tensor,
        loss_fn: Loss,
        optimizer: optim.Optimizer,
    ) -> Tuple[Tensor, float]:
        output = model(inputs)
        loss = loss_fn(output, targets)
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
            for sample in islice(val_loader, self.val_batches):
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
        accuracy = cdtm.accuracy(y_pred=preds, y_true=group_ids).item()
        wandb.log({"accuracy": accuracy}, step=step)
        return accuracy
