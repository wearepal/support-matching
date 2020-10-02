from pathlib import Path

print("starting up...")  # print this before loading all those heavy libraries

import typer
from sklearn.metrics import confusion_matrix

from clustering.optimisation.utils import get_class_id
from shared.utils import ClusterResults, load_results, save_results


def main(
    s_labels: Path = typer.Option(...),
    y_labels: Path = typer.Option(...),
    merged_labels: Path = typer.Option(...),
) -> None:
    class _ForYLabel:
        cluster_label_file = str(y_labels)

    class _ForSLabel:
        cluster_label_file = str(s_labels)

    print(f"Loading from {y_labels}")
    y_results = load_results(_ForYLabel(), check=False)
    print(f"Loading from {s_labels}")
    s_results = load_results(_ForSLabel(), check=False)
    s_count = int(s_results.class_ids.max() + 1)
    print(f"Computed s_count={s_count}")
    cluster_ids = get_class_id(
        s=s_results.cluster_ids, y=y_results.cluster_ids, s_count=s_count, to_cluster="both"
    )
    class_ids = get_class_id(
        s=s_results.class_ids, y=y_results.class_ids, s_count=s_count, to_cluster="both"
    )

    accuracy = (class_ids == cluster_ids).float().mean()
    print(f"accuracy = {accuracy}")
    conf_mat = confusion_matrix(class_ids, cluster_ids, normalize="all")
    print(conf_mat)
    test_metrics = {}
    if y_results.test_metrics is not None:
        test_metrics.update({f"Y {k}": v for k, v in y_results.test_metrics.items()})
    if s_results.test_metrics is not None:
        test_metrics.update({f"S {k}": v for k, v in s_results.test_metrics.items()})
    context_metrics = {}
    if y_results.context_metrics is not None:
        context_metrics.update({f"Y {k}": v for k, v in y_results.context_metrics.items()})
    if s_results.context_metrics is not None:
        context_metrics.update({f"S {k}": v for k, v in s_results.context_metrics.items()})

    cluster_results = ClusterResults(
        flags=y_results.flags,
        cluster_ids=cluster_ids,
        class_ids=class_ids,
        enc_path=y_results.enc_path,
        test_metrics=test_metrics,
        context_metrics=context_metrics,
    )
    save_results(save_path=merged_labels, cluster_results=cluster_results)


if __name__ == "__main__":
    typer.run(main)
