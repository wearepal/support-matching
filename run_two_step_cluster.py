"""Call the main functions of both parts one after the other."""
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory

assert sys.version_info >= (3, 8), f"please use Python 3.8 (this is 3.{sys.version_info.minor})"


def main():
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    raw_args = sys.argv[1:]
    if not all(
        (not arg.startswith("--")) or arg.startswith(("--c-", "--d-", "--b-")) for arg in raw_args
    ):
        print(
            "\nUse --b- to prefix those flags that will be passed to both parts of the code.\n"
            "Use --c- to prefix those flags that will only be passed to the clustering code.\n"
            "Use --d- to prefix those flags that will only be passed to the disentangling code.\n"
            "So, for example: --b-dataset cmnist --c-epochs 100"
        )
        raise RuntimeError("all flags have to use the prefix '--b-', '--c-' or '--d-'.")
    tmpdir = Path() / "test1"
    if True:
        clust_args = [arg.replace("--c-", "--").replace("--b-", "--") for arg in raw_args]

        # cluster into y
        clf_y = str(Path(tmpdir) / "labels_y.pth")
        clf_y_flag = ["--cluster-label-file", clf_y, "--cluster", "y"]
        run([sys.executable, "run_cl.py"] + clust_args + clf_y_flag, check=True)

        # cluster into s
        clf_s = str(Path(tmpdir) / "labels_s.pth")
        clf_s_flag = ["--cluster-label-file", clf_s, "--cluster", "s"]
        run([sys.executable, "run_cl.py"] + clust_args + clf_s_flag, check=True)

        # merge the two cluster label files
        clf_merged = str(Path(tmpdir) / "class_ids.pth")
        merge_args = ["--s-labels", clf_s, "--y-labels", clf_y, "--merged-labels", clf_merged]
        run([sys.executable, "merge_cluster_label_files.py"] + merge_args, check=True)

        # disentangling
        clf_both_flag = ["--cluster-label-file", clf_merged]
        dis_args = [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args]
        run([sys.executable, "run_d.py"] + dis_args + clf_both_flag, check=True)


if __name__ == "__main__":
    main()
