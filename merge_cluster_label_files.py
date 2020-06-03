from pathlib import Path

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
    y_labels, flags = load_results(_ForYLabel(), check=False)
    print(f"Loading from {args.s_labels}")
    s_labels, _ = load_results(_ForSLabel(), check=False)
    s_count = s_labels.max() + 1
    print(f"Computed s_count={s_count}")
    class_ids = get_class_id(s=s_labels, y=y_labels, s_count=s_count, to_cluster="both")

    class _ForMerged:
        cluster_label_file = str(args.merged_labels)

        @staticmethod
        def as_dict():
            return flags

    save_results(_ForMerged(), class_ids, results_dir=Path())


if __name__ == "__main__":
    main()
