from pathlib import Path

import ethicml as em
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from shared.configs import DS, BaseArgs
from shared.data import DatasetTriplet, get_data_tuples, load_dataset
from shared.utils import compute_metrics, make_tuple_from_data

cs = ConfigStore.instance()
cs.store(name="logistic_regression", node=BaseArgs)


@hydra.main(config_path="conf", config_name="logistic_regression")
def baseline_metrics(cfg: BaseArgs) -> None:
    assert cfg.data.dataset == DS.adult, "This script is only for the adult dataset."
    data: DatasetTriplet = load_dataset(cfg)
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
            cfg=cfg,
            predictions=preds,
            actual=test_data,
            exp_name="baseline",
            model_name=clf.name,
            step=0,
            save_to_csv=Path(to_absolute_path(cfg.misc.save_dir)),
            results_csv=cfg.misc.results_csv,
        )


if __name__ == "__main__":
    baseline_metrics()
