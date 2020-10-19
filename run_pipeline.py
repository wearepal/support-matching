"""Call the main functions of both parts one after the other."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import neptune

from shared.utils.flag_prefixes import accept_prefixes


def main() -> None:
    """First run the clustering, then pass on the cluster labels to the fair representation code."""
    # find out whether logging was turned on
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging", default=True, type=eval, choices=[True, False])
    dis_args = accept_prefixes(("--a-", "--d-", "--e-"))
    temp_args, _ = parser.parse_known_args(dis_args)
    if temp_args.logging:
        neptune.init("pal/fdm")

    with TemporaryDirectory() as tmpdir:
        clf = Path(tmpdir) / "labels.pth"

        from clustering.optimisation import main as cluster

        cluster(cluster_label_file=clf, logging=False)

        from fdm.optimisation import main as disentangle

        disentangle(cluster_label_file=clf, initialize_logging=False)


if __name__ == "__main__":
    main()
