from pathlib import Path

import ethicml as em

from shared.configs import BaseArgs
from shared.data import DatasetTriplet, get_data_tuples, load_dataset
from shared.utils import accept_prefixes, compute_metrics, confirm_empty, make_tuple_from_data


class BaselineArgs(BaseArgs):
    save_dir: Path = Path(".") / "experiments" / "finn"
    results_csv: str

    def add_arguments(self) -> None:
        super().add_arguments()
        self.add_argument("--save-dir", type=Path)


def baseline_metrics(args: BaselineArgs) -> None:
    assert args.dataset == "adult", "This script is only for the adult dataset."
    data: DatasetTriplet = load_dataset(args)
    train_data = data.train
    test_data = data.test
    if not isinstance(train_data, em.DataTuple):
        train_data, test_data = get_data_tuples(train_data, test_data)

    train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=False)

    for clf in [
        em.SVM(kernel="linear"),
        em.SVM(),
        em.Majority(),
        em.Kamiran(classifier="LR"),
        em.LRCV(),
        em.LR(),
    ]:
        preds = clf.run(train_data, test_data)

        compute_metrics(
            args=args,
            predictions=preds,
            actual=test_data,
            exp_name="baseline",
            model_name=clf.name,
            step=0,
            save_to_csv=args.save_dir,
            results_csv=args.results_csv,
        )


if __name__ == "__main__":
    args = BaselineArgs(fromfile_prefix_chars="@", explicit_bool=True, underscores_to_dashes=True)
    args.parse_args(accept_prefixes(("--a-", "--b-")), known_only=True)
    confirm_empty(args.extra_args, to_ignore=("--c-", "--d-", "--e-"))
    print(args)
    baseline_metrics(args=args)
