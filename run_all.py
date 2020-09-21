"""Call the main functions of both parts one after the other."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import wandb


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

    clust_args = [arg.replace("--c-", "--").replace("--b-", "--") for arg in raw_args]
    dis_args = [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args]

    # find out whether wandb was turned on
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", default=True, type=eval, choices=[True, False])
    temp_args, _ = parser.parse_known_args(dis_args)
    if temp_args.use_wandb:
        wandb.init(entity="predictive-analytics-lab", project="fdm")

    with TemporaryDirectory() as tmpdir:
        clf = str(Path(tmpdir) / "labels.pth")
        clf_flag = ["--cluster-label-file", clf]
        from clustering.optimisation import main as cluster

        cluster(clust_args + clf_flag + ["--use-wandb", "False"], known_only=True)
        from fdm.optimisation import main as disentangle

        disentangle(dis_args + clf_flag, known_only=True, initialize_wandb=False)


if __name__ == "__main__":
    main()
