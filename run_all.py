"""Call the main functions of both parts one after the other."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

assert sys.version_info >= (3, 8), f"please use Python 3.8 (this is 3.{sys.version_info.minor})"


def main() -> None:
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
        clust_args = [arg.replace("--c-", "--").replace("--b-", "--") for arg in raw_args]
        from clustering.optimisation import main as cluster

        cluster(clust_args + clf_flag + ["--use-wandb", "False"], known_only=True)
        dis_args = [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args]
        from fdm.optimisation import main as disentangle

        disentangle(dis_args + clf_flag, known_only=True)


if __name__ == "__main__":
    main()