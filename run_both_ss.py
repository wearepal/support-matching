"""Call the main functions of both parts one after the other."""
from pathlib import Path
import shlex
from subprocess import CalledProcessError, run
import sys
from tempfile import TemporaryDirectory


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the semi-supervised algorithm."""
    raw_args = sys.argv[1:]
    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = [f"misc.cluster_label_file={clf}"]
        try:
            run([sys.executable, "run_clust.py"] + raw_args + clf_flag, check=True)
            run([sys.executable, "run_ss.py"] + raw_args + clf_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
