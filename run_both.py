"""Call the main functions of both parts one after the other."""
from pathlib import Path
import shlex
from subprocess import run, CalledProcessError
import sys
from tempfile import TemporaryDirectory


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    raw_args = sys.argv[1:]
    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = ["--cluster-label-file", clf]
        try:
            run([sys.executable, "run_cl.py"] + raw_args + clf_flag, check=True)
            run([sys.executable, "run_d.py"] + raw_args + clf_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
