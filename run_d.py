"""Simply call the main function"""
import sys

assert sys.version_info >= (3, 8), f"please use Python 3.8 (this is 3.{sys.version_info.minor})"
from fdm.optimisation import main

if __name__ == "__main__":
    main(known_only=False)
