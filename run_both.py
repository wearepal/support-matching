"""Call the main functions of both parts one after the other."""
from pathlib import Path
import shlex
from subprocess import run, CalledProcessError
import sys
from tempfile import TemporaryDirectory

from shared.utils.flag_prefixes import accept_prefixes, check_args


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    raw_args = check_args()
    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = ["--cluster-label-file", clf]
        try:
            clust_args = accept_prefixes(raw_args, ("--a-", "--c-", "--e-"))
            run([sys.executable, "unsafe_run_cl.py"] + clust_args + clf_flag, check=True)
            dis_args = accept_prefixes(raw_args, ("--a-", "--d-", "--e-"))
            run([sys.executable, "unsafe_run_d.py"] + dis_args + clf_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
