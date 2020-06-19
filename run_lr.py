from __future__ import annotations

from pathlib import Path

from ethicml.algorithms import inprocess as algos  # type: ignore[misc]
from ethicml.utility import DataTuple  # type: ignore[misc]
from fdm.optimisation.evaluation import compute_metrics, make_tuple_from_data  # type: ignore[misc]
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

    for clf in [
        algos.SVM(kernel="linear"),
        algos.SVM(),
        algos.Majority(),
        algos.Kamiran(classifier="LR"),
        algos.LRCV(),
        algos.LR(),
    ]:
        preds = clf.run(train_data, test_data)

        compute_metrics(
            args=args,
            predictions=preds,
            actual=test_data,
            data_exp_name="baseline",
            model_name=clf.name,
            step=0,
            save_to_csv=args.save_dir,
            results_csv=args.results_csv,
        )


if __name__ == "__main__":
    args = BaselineArgs()
    args.parse_args()
    baseline_metrics(args)
