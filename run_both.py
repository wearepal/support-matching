"""Call the main functions of both parts one after the other."""
from pathlib import Path
import shlex
from subprocess import run, CalledProcessError
import sys
from tempfile import TemporaryDirectory


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
    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = ["--cluster-label-file", clf]
        try:
            clust_args = [arg.replace("--c-", "--").replace("--b-", "--") for arg in raw_args]
            run([sys.executable, "unsafe_run_cl.py"] + clust_args + clf_flag, check=True)
            dis_args = [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args]
            run([sys.executable, "unsafe_run_d.py"] + dis_args + clf_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
