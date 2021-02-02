from pathlib import Path

import ethicml as em
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from shared.configs import AdultConfig, BaseConfig, register_configs
from shared.data import DatasetTriplet, get_data_tuples, load_dataset
from shared.utils import compute_metrics, make_tuple_from_data, write_results_to_csv

cs = ConfigStore.instance()
cs.store(name="logistic_regression_schema", node=BaseConfig)
register_configs()


@hydra.main(config_path="conf", config_name="logistic_regression")
def baseline_metrics(hydra_config: DictConfig) -> None:
    cfg = BaseConfig.from_hydra(hydra_config)
    assert isinstance(cfg.data, AdultConfig), "This script is only for the adult dataset."
    data: DatasetTriplet = load_dataset(cfg)
    train_data = data.train
    test_data = data.test
    if not isinstance(train_data, em.DataTuple):
        train_data, test_data = get_data_tuples(train_data, test_data)

    train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=False)

    all_metrics = {}
    for clf in [
        em.SVM(kernel="linear"),
        em.SVM(),
        em.Majority(),
        em.Kamiran(classifier="LR"),
        em.LRCV(),
        em.LR(),
    ]:
        preds = clf.run(train_data, test_data)

        metrics = compute_metrics(
            cfg=cfg,
            predictions=preds,
            actual=test_data,
            model_name=clf.name,
            step=0,
            s_dim=data.s_dim,
        )
        all_metrics.update(metrics)

    cfg.misc.log_method = "baseline"
    write_results_to_csv(
        cfg,
        results=all_metrics,
        csv_dir=Path(to_absolute_path(cfg.misc.save_dir)),
        csv_file="baseline_" + cfg.misc.results_csv,
    )


if __name__ == "__main__":
    baseline_metrics()
