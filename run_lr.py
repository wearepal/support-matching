from pathlib import Path

import ethicml as em
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd

from shared.configs import BaseConfig, register_configs
from shared.data import DataModule
from shared.utils import (
    as_pretty_dict,
    compute_metrics,
    flatten_dict,
    make_tuple_from_data,
    write_results_to_csv,
)

cs = ConfigStore.instance()
cs.store(name="logistic_regression_schema", node=BaseConfig)
register_configs()


@hydra.main(config_path="conf", config_name="logistic_regression")
def baseline_metrics(hydra_config: DictConfig) -> None:
    cfg = BaseConfig.from_hydra(hydra_config)
    cfg_dict = flatten_dict(as_pretty_dict(cfg))
    data = DataModule.from_configs(cfg)
    train_data = data.train
    test_data = data.test
    train_data, test_data = get_data_tuples(train_data, test_data)
    data = [pd.DataFrame(tensor.detach().cpu().numpy()) for tensor in data]
    if sens_attrs is not None:
        data[1].columns = sens_attrs
    # create a DataTuple
    return em.DataTuple(x=data[0], s=data[1], y=data[2])

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
            predictions=preds,
            actual=test_data,
            model_name=clf.name,
            step=0,
            s_dim=data.dim_s,
        )
        all_metrics.update(metrics)

    cfg.train.log_method = "ethicml"
    results = {}
    results.update(cfg_dict)
    results.update(all_metrics)
    write_results_to_csv(
        results=results,
        csv_dir=Path(to_absolute_path(cfg.train.save_dir)),
        csv_file=cfg.datamodule.log_name + "_baseline_" + cfg.train.results_csv,
    )


if __name__ == "__main__":
    baseline_metrics()
