from itertools import islice
from typing import Final, List, Tuple

import clip
from conduit import metrics as cdtm
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
from conduit.types import Loss
from data_loading import CLIP_VER, DOWNLOAD_ROOT, MODEL_PATH, get_data
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor, nn, optim
from tqdm import tqdm
import wandb

from shared.data.utils import labels_to_group_id

BATCH_SIZE: Final = 10
NUM_ITERS: Final = 2000
EVAL_STEPS: Final = 100
NUM_EVAL: Final = 400
S_COUNT: Final = 2
Y_COUNT: Final = 2
LR: Final = 1e-5
NUM_WORKERS: Final = 10
GPU: int = 0


def main() -> None:
    print("Loading CLIP model (downloading if needed)...", flush=True)
    model, transforms = clip.load(name=CLIP_VER.value, device="cpu", download_root=DOWNLOAD_ROOT)
    print("Done.")
    use_gpu = torch.cuda.is_available() and GPU >= 0
    device = torch.device(f"cuda:{GPU}" if use_gpu else "cpu")
    visual_model = model.visual
    model = nn.Sequential(visual_model, nn.Linear(visual_model.output_dim, S_COUNT * Y_COUNT))
    model.to(device)
    dm = get_data(transforms, batch_size_tr=BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="clip-finetuning",
        config={
            "clip_version": CLIP_VER,
            "batch_size": BATCH_SIZE,
            "num_iters": NUM_ITERS,
            "learning_rate": LR,
        },
    )
    assert run is not None
    train(
        model,
        train_data=dm.train_dataloader(balance=True, num_workers=NUM_WORKERS),
        eval_data=dm.test_dataloader(),
        optimizer=optimizer,
        iters=NUM_ITERS,
        device=device,
    )
    torch.save(model[0].state_dict(), MODEL_PATH)
    run.finish()


def train(
    model: nn.Module,
    train_data: CdtDataLoader[TernarySample[Tensor]],
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    optimizer: optim.Optimizer,
    iters: int,
    device: torch.device,
) -> None:
    model.train()
    pbar = tqdm(islice(train_data, iters), total=iters)
    last_acc = 0.0
    loss_fn = CrossEntropyLoss(reduction="mean")
    for step, sample in enumerate(pbar, start=1):
        x = sample.x.to(device, non_blocking=True)
        s = sample.s.to(device, non_blocking=True)
        y = sample.y.to(device, non_blocking=True)
        group_id = labels_to_group_id(s=s, y=y, s_count=S_COUNT)

        _, loss = generic_train_step(
            model, inputs=x, targets=group_id, loss_fn=loss_fn, optimizer=optimizer
        )

        to_log = {
            "loss": loss,
            "s1_share": sample.s.float().mean().item(),
            "y1_share": sample.y.float().mean().item(),
        }
        wandb.log(to_log, step=step)
        if step % EVAL_STEPS == 0:
            last_acc = eval(model, eval_data, device, step)
        pbar.set_postfix(loss=loss, last_acc=last_acc)


def generic_train_step(
    model: nn.Module, *, inputs: Tensor, targets: Tensor, loss_fn: Loss, optimizer: optim.Optimizer
) -> Tuple[Tensor, float]:
    output = model(inputs)
    loss = loss_fn(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def eval(
    model: nn.Module,
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    device: torch.device,
    step: int,
) -> float:
    preds, s, y = predict(model, eval_data, device)
    group_ids = labels_to_group_id(s=s, y=y, s_count=2)
    accuracy = cdtm.accuracy(y_pred=preds, y_true=group_ids).item()
    wandb.log({"accuracy": accuracy}, step=step)
    return accuracy


def predict(
    model: nn.Module,
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    model.eval()
    all_preds: List[Tensor] = []
    all_s: List[Tensor] = []
    all_y: List[Tensor] = []
    with torch.no_grad():
        for sample in islice(eval_data, NUM_EVAL):
            logits = model(sample.x.to(device, non_blocking=True))
            all_preds.append(torch.argmax(logits, dim=-1).detach().cpu())
            all_s.append(sample.s)
            all_y.append(sample.y)

    preds = torch.cat(all_preds)
    s = torch.cat(all_s)
    y = torch.cat(all_y)
    return preds, s, y


if __name__ == "__main__":
    main()
