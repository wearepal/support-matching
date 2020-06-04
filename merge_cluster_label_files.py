from pathlib import Path

from sklearn.metrics import confusion_matrix
from typed_flags import TypedFlags

from clustering.optimisation.utils import get_class_id
from shared.utils import load_results, save_results


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
    y_pred, y_true, flags = load_results(_ForYLabel(), check=False)
    print(f"Loading from {args.s_labels}")
    s_pred, s_true, _ = load_results(_ForSLabel(), check=False)
    s_count = int(s_true.max() + 1)
    print(f"Computed s_count={s_count}")
    cluster_ids = get_class_id(s=s_pred, y=y_pred, s_count=s_count, to_cluster="both")
    class_ids = get_class_id(s=s_true, y=y_true, s_count=s_count, to_cluster="both")

    accuracy = (class_ids == cluster_ids).float().mean()
    print(f"accuracy = {accuracy}")
    conf_mat = confusion_matrix(class_ids, cluster_ids, normalize="all")
    print(conf_mat)

    save_results(
        cluster_ids=cluster_ids, class_ids=class_ids, save_path=args.merged_labels, flags=flags
    )


if __name__ == "__main__":
    main()
