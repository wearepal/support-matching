from pathlib import Path

from sklearn.metrics import confusion_matrix
from typed_flags import TypedFlags

from clustering.optimisation.utils import get_class_id
from shared.utils import ClusterResults, load_results, save_results


class Args(TypedFlags):
    s_labels: Path
    y_labels: Path
    merged_labels: Path


def main():
    args = Args().parse_args()

    class _ForYLabel:
        cluster_label_file = str(args.y_labels)

    class _ForSLabel:
        cluster_label_file = str(args.s_labels)

    print(f"Loading from {args.y_labels}")
    y_results = load_results(_ForYLabel(), check=False)
    print(f"Loading from {args.s_labels}")
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

    cluster_results = ClusterResults(
        flags=y_results.flags,
        cluster_ids=cluster_ids,
        class_ids=class_ids,
        enc_path=y_results.enc_path,
        test_acc=0.5 * (y_results.test_acc + s_results.test_acc),
        context_acc=0.5 * (y_results.context_acc + s_results.context_acc),
    )
    save_results(save_path=args.merged_labels, cluster_results=cluster_results)


if __name__ == "__main__":
    main()
