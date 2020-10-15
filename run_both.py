"""Call the main functions of both parts one after the other."""
import os
import shlex
import sys
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    raw_args = sys.argv[1:]
    env = os.environ
    if env.get("STARTED_BY_GUILDAI", None) == "1":
        src = Path(env["GUILD_SOURCECODE"])
    else:
        src = Path(".")
    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = ["--cluster-label-file", clf]
        try:
            run([sys.executable, str(src / "run_clust.py")] + raw_args + clf_flag, check=True)
            run([sys.executable, str(src / "run_dis.py")] + raw_args + clf_flag, check=True)
        except CalledProcessError as cpe:
            # catching the exception ourselves leads to much nicer error messages
            print(f"\nCommand '{shlex.join(cpe.cmd)}'")


if __name__ == "__main__":
    main()
