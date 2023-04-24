from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict
from typing_extensions import Self, TypeAlias

from conduit import metrics as cdtm
from conduit.models.utils import prefix_keys
import ethicml as em
import ethicml.metrics as emm
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import wandb

from src.data import EvalTuple
from src.utils import to_numpy

__all__ = [
    "MetricDict",
    "SummaryMetric",
    "compute_metrics",
    "print_metrics",
    "write_results_to_csv",
]


@dataclass
class EmEvalPair:
    pred: em.Prediction
    actual: em.LabelTuple

    @classmethod
    def from_et(cls, et: EvalTuple, *, pred_s: bool = False) -> Self:
        return cls.from_tensors(y_pred=et.y_pred, y_true=et.y_true, s=et.s, pred_s=pred_s)

    @classmethod
    def from_tensors(
        cls, y_pred: Tensor, *, y_true: Tensor, s: Tensor, pred_s: bool = False
    ) -> Self:
        if len(y_pred) != len(y_true) != len(s):
            raise ValueError("'y_pred', 'y_true', and 's' must match in size at dimension 0.")
        pred = em.Prediction(hard=pd.Series(to_numpy(y_pred.flatten())))
        sens_pd = pd.Series(to_numpy(tensor=s.flatten()).astype(np.float32), name="subgroup")
        labels_pd = pd.Series(to_numpy(y_true.flatten()), name="labels")
        actual = em.LabelTuple.from_df(s=sens_pd, y=sens_pd if pred_s else labels_pd)
        return cls(pred=pred, actual=actual)


MetricDict: TypeAlias = Dict[str, float]


class SummaryMetric(Enum):
    ACC = "Accuracy"
    ROB_ACC = "Robust_Accuracy"
    BAL_ACC = "Balanced_Accuracy"
    ROB_GAP = "Robust_Gap"
    TPR = "TPR"
    TNR = "TNR"
    ROB_TPR = "Robust_TPR"
    ROB_TNR = "Robust_TNR"
    ROB_TPR_GAP = "Robust_TPR_Gap"
    ROB_TNR_GAP = "Robust_TNR_Gap"
    RENYI = "Renyi preds and s"


robust_tpr_gap = cdtm.subclasswise_metric(
    comparator=partial(cdtm.conditional_equal, y_true_cond=1), aggregator=cdtm.Aggregator.MAX_DIFF
)
robust_tnr_gap = cdtm.subclasswise_metric(
    comparator=partial(cdtm.conditional_equal, y_true_cond=0), aggregator=cdtm.Aggregator.MAX_DIFF
)


@torch.no_grad()
def compute_metrics(
    pair: EmEvalPair,
    *,
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
    metrics = emm.run_metrics(
        predictions=predictions,
        actual=actual,
        metrics=[emm.Accuracy(), emm.TPR(), emm.TNR(), emm.RenyiCorrelation()],
        per_sens_metrics=[emm.Accuracy(), emm.ProbPos(), emm.TPR(), emm.TNR()],
    )
    # Convert to tensor for compatibility with conduit-derived metrics.
    y_pred_t = torch.as_tensor(torch.as_tensor(predictions.hard, dtype=torch.long))
    y_true_t = torch.as_tensor(torch.as_tensor(actual.y, dtype=torch.long))
    s_t = torch.as_tensor(torch.as_tensor(actual.s, dtype=torch.long))
    cdt_metrics = {
        SummaryMetric.ROB_ACC.value: cdtm.robust_accuracy,
        SummaryMetric.BAL_ACC.value: cdtm.subclass_balanced_accuracy,
        SummaryMetric.ROB_GAP.value: cdtm.robust_gap,
        SummaryMetric.ROB_TPR_GAP.value: robust_tpr_gap,
        SummaryMetric.ROB_TNR_GAP.value: robust_tnr_gap,
        SummaryMetric.ROB_TPR.value: cdtm.robust_tpr,
        SummaryMetric.ROB_TNR.value: cdtm.robust_tnr,
    }
    for name, fn in cdt_metrics.items():
        metrics[name] = fn(y_pred=y_pred_t, y_true=y_true_t, s=s_t).item()
    # replace the slash; it's causing problems
    metrics = {k.replace("/", "÷"): v for k, v in metrics.items()}
    if exp_name:
        metrics = prefix_keys(metrics, prefix=exp_name, sep="/")
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
        logger.info(f"Results for {exp_name or ''}:")
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
    results_df = pd.DataFrame(to_log, index=[0])

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
