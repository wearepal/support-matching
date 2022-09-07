from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing_extensions import Self, TypeAlias

from conduit import metrics as cdtm
from conduit.models.utils import prefix_keys
import ethicml as em
import ethicml.metrics as emm
from ethicml.run import run_metrics
from ethicml.utility.data_structures import LabelTuple
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import wandb

from src.utils import to_numpy

__all__ = [
    "MetricDict",
    "compute_metrics",
    "print_metrics",
    "write_results_to_csv",
]


@dataclass
class EmEvalPair:
    pred: em.Prediction
    actual: LabelTuple

    @classmethod
    def from_tensors(
        cls, y_pred: Tensor, *, y_true: Tensor, s: Tensor, pred_s: bool = False
    ) -> Self:
        if len(y_pred) != len(y_true) != len(s):
            raise ValueError("'y_pred', 'y_true', and 's' must match in size at dimension 0.")
        pred = em.Prediction(hard=pd.Series(to_numpy(y_pred.flatten())))
        sens_pd = pd.Series(to_numpy(tensor=s.flatten()).astype(np.float32), name="subgroup")
        labels_pd = pd.Series(to_numpy(y_true.flatten()), name="labels")
        actual = LabelTuple.from_df(s=sens_pd, y=sens_pd if pred_s else labels_pd)
        return cls(pred=pred, actual=actual)


MetricDict: TypeAlias = Dict[str, float]

@torch.no_grad()
def compute_metrics(
    pair: EmEvalPair,
    *,
    model_name: str,
    step: int | None = None,
    exp_name: str | None = None,
    save_summary: bool = False,
    use_wandb: bool = False,
    additional_entries: Mapping[str, float] | None = None,
    prefix: str | None = None,
    verbose: bool = True,
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
    logger.info("Computing classification metrics")
    predictions = pair.pred
    actual = pair.actual
    predictions._info = {}  # type: ignore
    metrics = run_metrics(
        predictions=predictions,
        actual=actual,
        metrics=[emm.Accuracy(), emm.TPR(), emm.TNR(), emm.RenyiCorrelation()],  # type: ignore
        per_sens_metrics=[emm.Accuracy(), emm.ProbPos(), emm.TPR(), emm.TNR()],  # type: ignore
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

    if verbose:
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
