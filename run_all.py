"""Call the main functions of both parts one after the other."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import wandb

from shared.utils.flag_prefixes import check_args, accept_prefixes


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    raw_args = check_args()

    clust_args = accept_prefixes(raw_args, ("--a-", "--c-", "--e-"))
    dis_args = accept_prefixes(raw_args, ("--a-", "--d-", "--e-"))

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
