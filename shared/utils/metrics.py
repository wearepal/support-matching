from __future__ import annotations
from collections.abc import Mapping
import logging
from pathlib import Path
from typing import List, TypeVar

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
    "accuracy_per_subgroup",
    "compute_metrics",
    "make_tuple_from_data",
    "print_metrics",
    "robust_accuracy",
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


T = TypeVar("T", Tensor, npt.NDArray[np.integer])


def accuracy_per_subgroup(y_pred: T, *, y_true: T, s: T) -> List[float]:
    unique_fn = torch.unique if isinstance(y_pred, Tensor) else np.unique
    s_unique, s_counts = unique_fn(s, return_counts=True)
    s_m = s.flatten()[:, None] == s_unique[None]
    hits = y_pred.flatten() == y_true.flatten()
    return ((hits[:, None] * s_m).sum(0) / s_counts).tolist()


def robust_accuracy(y_pred: T, *, y_true: T, s: T) -> float:
    return min(accuracy_per_subgroup(y_pred=y_pred, y_true=y_true, s=s))


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
    metrics["Robust_Accuracy"] = robust_accuracy(
        y_pred=predictions.hard.to_numpy(),
        y_true=actual.y.to_numpy(),
        s=actual.s.to_numpy(),
    )
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
