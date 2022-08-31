from __future__ import annotations
import numpy as np
from torch import Tensor
from typing_extensions import Self
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path

from conduit import metrics as cdtm
from conduit.models.utils import prefix_keys
import ethicml as em
import ethicml.metrics as emm
from ethicml.utility.data_structures import LabelTuple
from loguru import logger
import pandas as pd
import torch
import wandb

__all__ = [
    "compute_metrics",
    "print_metrics",
    "write_results_to_csv",
]

@dataclass
class EvalPair:
    pred: em.Prediction
    actual: LabelTuple

    @classmethod
    def from_tensors(cls, y_pred: Tensor, *, y_true: Tensor, s: Tensor, pred_s: bool = False) -> Self:
        pred = em.Prediction(hard=pd.Series(y_pred))
        sens_pd = pd.Series(s.detach().cpu().numpy().astype(np.float32), name="subgroup")
        labels_pd = pd.Series(y_true.detach().cpu().numpy(), name="labels")
        actual = LabelTuple.from_df(s=sens_pd, y=sens_pd if pred_s else labels_pd)
        return cls(pred=pred, actual=actual)

@torch.no_grad()
def compute_metrics(
    pair: EvalPair,
    *,
    model_name: str,
    s_dim: int,
    step: int | None = None,
    exp_name: str | None = None,
    save_summary: bool = False,
    use_wandb: bool = False,
    additional_entries: Mapping[str, float] | None = None,
    prefix: str | None = None,
) -> dict[str, float]:
    """Compute accuracy and fairness metrics and log them.

    :param pair: predictions and labels in a format that is compatible with EthicML
    :param model_name: name of the model used
    :param step: step of training (needed for logging to W&B)
    :param s_dim: dimension of s
    :param exp_name: name of the experiment
    :param save_summary: if True, a summary will be saved to wandb
    :param use_wandb: whether to use wandb at all
    :param additional_entries: entries that should go with in the summary

    :returns: dictionary with the computed metrics
    """

    predictions = pair.pred
    actual = pair.actual
    predictions._info = {}  # type: ignore
    metrics = em.run_metrics(
        predictions=predictions,
        actual=actual,
        metrics=[emm.Accuracy(), emm.TPR(), emm.TNR(), emm.RenyiCorrelation()],  # type: ignore
        per_sens_metrics=[emm.Accuracy(), emm.ProbPos(), emm.TPR(), emm.TNR()],  # type: ignore
        diffs_and_ratios=s_dim < 4,  # this just gets too much with higher s dim
    )
    # Convert to tensor for compatibility with conduit-derived metrics.
    y_pred_t = torch.as_tensor(torch.as_tensor(predictions.hard, dtype=torch.long))
    y_true_t = torch.as_tensor(torch.as_tensor(actual.y, dtype=torch.long))
    s_t = torch.as_tensor(torch.as_tensor(actual.s, dtype=torch.long))
    cdt_metrics = {
        "Robust_Accuracy": cdtm.robust_accuracy,
        "Balanced_Accuracy": cdtm.subclass_balanced_accuracy,
        "Robust_TPR": cdtm.robust_tpr,
        "Robust_TNR": cdtm.robust_tnr,
    }
    for name, fn in cdt_metrics.items():
        metrics[name] = fn(y_pred=y_pred_t, y_true=y_true_t, s=s_t).item()
    # replace the slash; it's causing problems
    metrics = {k.replace("/", "÷"): v for k, v in metrics.items()}
    metrics = {f"{k} ({model_name})": v for k, v in metrics.items()}
    if exp_name:
        metrics = {f"{exp_name}/{k}": v for k, v in metrics.items()}
    if prefix is not None:
        metrics = prefix_keys(metrics, prefix=prefix, sep="/")

    if use_wandb:
        wandb.log(metrics, step=step)

        if save_summary:
            external = additional_entries or {}

            for metric_name, value in metrics.items():
                wandb.run.summary[metric_name] = value  # type: ignore
            for metric_name, value in external.items():
                wandb.run.summary[metric_name] = value  # type: ignore

    logger.info(f"Results for {exp_name or ''} ({model_name}):")
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: Mapping[str, int | float | str]) -> None:
    """Print metrics such that they don't clutter everything too much."""
    for key, value in metrics.items():
        logger.info(f"    {key}: {value:.3g}")
    logger.info("---")


def write_results_to_csv(results: Mapping[str, int | float | str], csv_dir: Path, csv_file: str):
    to_log = {}
    # to_log.update(flatten_dict(as_pretty_dict(cfg)))
    to_log.update(results)
    # I don't know why it has to be done in 2 steps, but it that's how it is
    results_df = pd.DataFrame(columns=list(to_log))
    results_df = results_df.append(to_log, ignore_index=True, sort=False)

    csv_dir.mkdir(exist_ok=True, parents=True)

    results_path = csv_dir / csv_file
    if results_path.exists():
        # load previous results and append new results
        previous_results = pd.read_csv(results_path)
        results_df = pd.concat(
            [previous_results, results_df], sort=False, ignore_index=True, axis="index"
        )
    results_df.reset_index(drop=True).to_csv(results_path, index=False)
    logger.info(f"Results have been written to {results_path.resolve()}")