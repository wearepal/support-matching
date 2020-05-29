"""Simply call the main function"""
from pathlib import Path
import sys

assert sys.version_info >= (3, 8), f"please use Python 3.8 (this is 3.{sys.version_info.minor})"
from fdm.optimisation import main as fdm
from clustering.optimisation import main as clustering

if __name__ == "__main__":
    cluster_label_file: Path
    # first run the clustering, then pass on the cluster labels to the fair representation code
    raw_args = sys.argv[1:]
    if not all(
        (not arg.startswith("--")) or arg.startswith(("--c-", "--d-", "--b-")) for arg in raw_args
    ):
        print(
            "\nUse --b- to prefix those flags that will be passed to both parts of the code.\n"
            "Use --c- to prefix those flags that will be passed to the clustering code.\n"
            "Use --d- to prefix those flags that will be passed to the disentangling code.\n"
            "So, for example: --b-dataset cmnist --c-epochs 100"
        )
        raise RuntimeError("all flags have to use the prefix '--b-', '--c-' or '--d-'.")
    _, cluster_label_file = clustering(
        [arg.replace("--c-", "--").replace("--b-", "--") for arg in raw_args], known_only=True
    )
    fdm(
        [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args],
        known_only=True,
        cluster_label_file=cluster_label_file,
    )
