"""Simply call the main function"""
from pathlib import Path
import sys

assert sys.version_info >= (3, 8), f"please use Python 3.8 (this is 3.{sys.version_info.minor})"
from fdm.optimisation import main as fdm
from clustering.optimisation import main as clustering

if __name__ == "__main__":
    cluster_label_file: Path
    # first run the clustering, then pass on the cluster labels to the fair representation code
    _, cluster_label_file = clustering()
    fdm(cluster_label_file=cluster_label_file)
