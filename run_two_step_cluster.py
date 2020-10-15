"""Call the main functions of both parts one after the other."""
import os
import shlex
import sys
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    args = sys.argv[1:]
    env = os.environ
    if env.get("STARTED_BY_GUILDAI", None) == "1":
        src = Path(env["GUILD_SOURCECODE"])
    else:
        src = Path(".")
    with TemporaryDirectory() as tmpdir:
        try:
            # cluster into y
            clf_y = str(Path(tmpdir) / "labels_y.pth")
            clf_y_flag = ["--cluster-label-file", clf_y, "--cluster", "y"]
            run([sys.executable, str(src / "run_clust.py")] + args + clf_y_flag, check=True)

            # cluster into s
            clf_s = str(Path(tmpdir) / "labels_s.pth")
            clf_s_flag = ["--cluster-label-file", clf_s, "--cluster", "s"]
            run([sys.executable, str(src / "run_clust.py")] + args + clf_s_flag, check=True)

            # merge the two cluster label files
            clf_merged = str(Path(tmpdir) / "class_ids.pth")
            merge_args = ["--s-labels", clf_s, "--y-labels", clf_y, "--merged-labels", clf_merged]
            run(
                [sys.executable, str(src / "merge_cluster_label_files.py")] + merge_args, check=True
            )

            # disentangling
            clf_both_flag = ["--cluster-label-file", clf_merged]
            run([sys.executable, str(src / "run_dis.py")] + args + clf_both_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
