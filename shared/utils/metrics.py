from __future__ import annotations
from collections.abc import Mapping
from functools import partial
import logging
from pathlib import Path
from typing import List, TypeVar

from conduit import metrics as cdtm
from conduit.models.utils import prefix_keys
import ethicml as em
import ethicml.metrics as emm
from ethicml.utility.data_structures import LabelTuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
import wandb

__all__ = [
    "compute_metrics",
    "make_tuple_from_data",
    "print_metrics",
    "write_results_to_csv",
]

log = logging.getLogger(__name__.split(".")[-1].upper())


def make_tuple_from_data(
    train: em.DataTuple, test: em.DataTuple, pred_s: bool
) -> tuple[em.DataTuple, em.DataTuple]:
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return em.DataTuple.from_df(x=train_x, s=train.s, y=train_y), em.DataTuple.from_df(
        x=test_x, s=test.s, y=test_y
    )


robust_tpr = cdtm.subclasswise_metric(
    comparator=partial(cdtm.conditional_equal, y_true_cond=1), aggregator=cdtm.Aggregator.MIN
)
robust_tnr = cdtm.subclasswise_metric(
    comparator=partial(cdtm.conditional_equal, y_true_cond=0), aggregator=cdtm.Aggregator.MIN
)


def compute_metrics(
    predictions: em.Prediction,
    *,
    actual: LabelTuple,
    model_name: str,
    step: int,
    s_dim: int,
    exp_name: str | None = None,
    save_summary: bool = False,
    use_wandb: bool = False,
    additional_entries: Mapping[str, float] | None = None,
    prefix: str | None = None,
) -> dict[str, float]:
    """Compute accuracy and fairness metrics and log them.

    :param predictions: predictions in a format that is compatible with EthicML
    :param actual: labels for the predictions
    :param model_name: name of the model used
    :param step: step of training (needed for logging to W&B)
    :param s_dim: dimension of s
    :param exp_name: name of the experiment
    :param save_summary: if True, a summary will be saved to wandb
    :param use_wandb: whether to use wandb at all
    :param additional_entries: entries that should go with in the summary

    :returns: dictionary with the computed metrics
    """

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
        "Balanced_Accuracy": cdtm.group_balanced_accuracy,
        "Robust_TPR": robust_tpr,
        "Robust_TNR": robust_tnr,
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

    log.info(f"Results for {exp_name or ''} ({model_name}):")
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: Mapping[str, int | float | str]) -> None:
    """Print metrics such that they don't clutter everything too much."""
    for key, value in metrics.items():
        log.info(f"    {key}: {value:.3g}")
    log.info("---")


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
    log.info(f"Results have been written to {results_path.resolve()}")
