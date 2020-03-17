"""Simply call the main function"""
import sys
from fdm.optimisation import main

if __name__ == "__main__":
    assert sys.version_info.major >= 3, "we don't support Python 2"
    assert sys.version_info.minor >= 8, "please use Python 3.8"
    main()
