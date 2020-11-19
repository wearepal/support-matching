import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import ethicml as em
import wandb

from shared.configs import BaseArgs

from .utils import wandb_log

__all__ = ["compute_metrics", "make_tuple_from_data", "print_metrics"]

log = logging.getLogger(__name__.split(".")[-1].upper())


def make_tuple_from_data(
    train: em.DataTuple, test: em.DataTuple, pred_s: bool
) -> Tuple[em.DataTuple, em.DataTuple]:
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return em.DataTuple(x=train_x, s=train.s, y=train_y), em.DataTuple(x=test_x, s=test.s, y=test_y)


def compute_metrics(
    cfg: BaseArgs,
    predictions: em.Prediction,
    actual: em.DataTuple,
    exp_name: str,
    model_name: str,
    step: int,
    save_to_csv: Optional[Path] = None,
    results_csv: str = "",
    use_wandb: bool = False,
    additional_entries: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them.

    Args:
        args: args object
        predictions: predictions in a format that is compatible with EthicML
        actual: labels for the predictions
        exp_name: name of the experiment
        model_name: name of the model used
        step: step of training (needed for logging to W&B)
        save_to_csv: if a path is given, the results are saved to a CSV file
        results_csv: name of the CSV file
    Returns:
        dictionary with the computed metrics
    """

    predictions._info = {}
    metrics = em.run_metrics(
        predictions,
        actual,
        metrics=[em.Accuracy(), em.TPR(), em.TNR(), em.RenyiCorrelation()],
        per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR(), em.TNR()],
        diffs_and_ratios=cfg.misc._s_dim < 4,  # this just gets too much with higher s dim
    )
    # replace the slash; it's causing problems
    metrics = {k.replace("/", "÷"): v for k, v in metrics.items()}

    if use_wandb:
        wandb_log(cfg.misc, {f"{k} ({model_name})": v for k, v in metrics.items()}, step=step)

    if save_to_csv is not None:
        # full_name = f"{args.dataset}_{exp_name}"
        # exp_name += "_s" if pred_s else "_y"
        # if hasattr(args, "eval_on_recon"):
        #     exp_name += "_on_recons" if args.eval_on_recon else "_on_encodings"

        manual_entries = {
            "seed": str(getattr(cfg.misc, "seed", cfg.misc.data_split_seed)),
            "data": exp_name,
            "method": f'"{model_name}"',
            "wandb_url": str(wandb.run.get_url()) if use_wandb and cfg.misc.use_wandb else "(None)",
        }

        external = additional_entries or {}

        if results_csv:
            assert isinstance(save_to_csv, Path)
            save_to_csv.mkdir(exist_ok=True, parents=True)
            results = {**metrics, **external}

            results_path = save_to_csv / f"{cfg.data.dataset.name}_{model_name}_{results_csv}"
            values = ",".join(list(manual_entries.values()) + [str(v) for v in results.values()])
            if not results_path.is_file():
                with results_path.open("w") as f:
                    # ========= header =========
                    f.write(",".join(list(manual_entries) + [str(k) for k in results]) + "\n")
                    f.write(values + "\n")
            else:
                with results_path.open("a") as f:  # append to existing file
                    f.write(values + "\n")
            log.info(f"Results have been written to {results_path.resolve()}")
        if use_wandb:
            for metric_name, value in metrics.items():
                wandb.run.summary[f"{model_name}_{metric_name}"] = value
            # external metrics are without prefix
            for metric_name, value in external.items():
                wandb.run.summary[metric_name] = value

    log.info(f"Results for {exp_name} ({model_name}):")
    print_metrics({f"{k} ({model_name})": v for k, v in metrics.items()})
    log.info("")  # empty line
    return metrics


def print_metrics(metrics: Mapping[str, Union[int, float, str]]) -> None:
    """Print metrics in such a way that they are picked up by guildai."""
    log.info("---")
    log.info(
        "\n".join(f"{key.replace(' ', '_').lower()}: {value:.5g}" for key, value in metrics.items())
    )
    log.info("---")
