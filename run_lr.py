from __future__ import annotations

from pathlib import Path
from typing import Dict

from ethicml.algorithms import inprocess as algos
from ethicml.evaluators import run_metrics
from ethicml.metrics import TNR, TPR, Accuracy, ProbPos, RenyiCorrelation
from ethicml.utility import DataTuple, Prediction
from shared.data import DatasetTriplet, get_data_tuples, load_dataset
from shared.configs import BaseArgs


class BaselineArgs(BaseArgs):
    save_dir: Path = Path(".") / "experiments" / "finn"
    results_csv: str


def baseline_metrics(args: BaselineArgs) -> None:
    assert args.dataset == "adult", "This script is only for the adult dataset."
    data: DatasetTriplet = load_dataset(args)
    train_data = data.train
    test_data = data.test
    if not isinstance(train_data, DataTuple):
        train_data, test_data = get_data_tuples(train_data, test_data)

    train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=False)

    # clf = algos.SVM(kernel="linear")
    # clf = algos.Majority()
    # clf = algos.Kamiran(classifier="LR")
    clf = algos.LR()
    preds = clf.run(train_data, test_data)

    actual = test_data
    name = "LR baseline"
    metrics = compute_metrics(preds, actual, name, run_all=args._y_dim == 1)
    save_to_csv = args.save_dir
    sweep_key = "seed"
    sweep_value = str(args.data_split_seed)
    results_path = save_to_csv / f"{name}_{args.results_csv}"
    value_list = ",".join([sweep_value] + [str(v) for v in metrics.values()])
    if not results_path.is_file():
        with results_path.open("w") as f:
            f.write(
                ",".join([sweep_key] + ["method"] + [str(k) for k in metrics.keys()]) + "\n"
            )  # header
            f.write(value_list + "\n")
    else:
        with results_path.open("a") as f:  # append to existing file
            f.write(value_list + "\n")
    print(f"Results have been written to \"{results_path.resolve()}\".")


def compute_metrics(predictions: Prediction, actual, name: str, run_all=False) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), TPR(), TNR(), RenyiCorrelation()],
            per_sens_metrics=[ProbPos(), TPR(), TNR()],
        )
    else:
        metrics = run_metrics(
            predictions, actual, metrics=[Accuracy(), RenyiCorrelation()], per_sens_metrics=[]
        )
    print(f"Results for {name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line
    return metrics


def make_tuple_from_data(train, test, pred_s):
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return (DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y))


if __name__ == "__main__":
    args = BaselineArgs()
    args.parse_args()
    baseline_metrics(args)
