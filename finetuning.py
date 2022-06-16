from itertools import islice
from typing import Final, List, Tuple

import clip
from conduit.data.datasets.utils import CdtDataLoader
from conduit.data.structures import TernarySample
import numpy.typing as npt
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from data_loading import CLIPVersion, DOWNLOAD_ROOT, get_data
from shared.data.utils import labels_to_group_id

CLIP_VER: Final = CLIPVersion.RN50x4
BATCH_SIZE: Final = 2
NUM_ITERS: Final = 1000
EVAL_STEPS: Final = 100
NUM_EVAL: Final = 200


def main() -> None:
    print("Loading CLIP model (downloading if needed)...", flush=True)
    model, transforms = clip.load(
        # name=CLIPVersion.ViT_L14.value, device="cpu", download_root=DOWNLOAD_ROOT
        name=CLIP_VER.value,
        device="cpu",
        download_root=DOWNLOAD_ROOT,
    )
    print("Done.")
    visual_model = model.visual
    device = torch.device("cuda:0")
    model = nn.Sequential(visual_model, nn.Linear(visual_model.output_dim, 4))
    model.to(device)
    dm = get_data(transforms, batch_size_tr=BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="clip-finetuning",
        config={"clip_ver": CLIP_VER, "batch_size": BATCH_SIZE, "num_iters": NUM_ITERS},
    )
    assert run is not None
    train(
        model,
        train_data=dm.train_dataloader(balance=False),
        eval_data=dm.test_dataloader(),
        optimizer=optimizer,
        iters=NUM_ITERS,
        device=device,
    )
    run.finish()


def train(
    model: nn.Module,
    train_data: CdtDataLoader[TernarySample[Tensor]],
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    optimizer: optim.Optimizer,
    iters: int,
    device: torch.device,
):
    model.train()
    pbar = tqdm(islice(train_data, iters), total=iters)
    last_acc = 0.0
    for step, sample in enumerate(pbar, start=1):
        x = sample.x.to(device, non_blocking=True)
        s = sample.s.to(device, non_blocking=True)
        y = sample.y.to(device, non_blocking=True)
        group_id = labels_to_group_id(s=s, y=y, s_count=2)

        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, group_id)
        loss.backward()
        optimizer.step()

        to_log = {
            "loss": loss.item(),
            "s1_share": sample.s.float().mean().item(),
            "y1_share": sample.y.float().mean().item(),
        }
        wandb.log(to_log, step=step)
        if step % EVAL_STEPS == 0:
            last_acc = eval(model, eval_data, device, step)
        pbar.set_postfix(loss=loss.item(), acc=last_acc)


def eval(
    model: nn.Module,
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    device: torch.device,
    step: int,
) -> float:
    preds, s, y = predict(model, eval_data, device)
    group_ids = labels_to_group_id(s=s, y=y, s_count=2)
    accuracy = accuracy_score(group_ids, preds)
    wandb.log({"accuracy": accuracy}, step=step)
    return accuracy


def predict(
    model: nn.Module,
    eval_data: CdtDataLoader[TernarySample[Tensor]],
    device: torch.device,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    all_preds: List[Tensor] = []
    all_s: List[Tensor] = []
    all_y: List[Tensor] = []
    with torch.set_grad_enabled(False):
        for sample in islice(eval_data, NUM_EVAL):
            logits = model(sample.x.to(device, non_blocking=True))
            all_preds.append(torch.argmax(logits, dim=-1).detach().cpu())
            all_s.append(sample.s)
            all_y.append(sample.y)

    preds = torch.cat(all_preds).numpy()
    s = torch.cat(all_s).numpy()
    y = torch.cat(all_y).numpy()
    return preds, s, y


if __name__ == "__main__":
    main()
