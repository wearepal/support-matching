import subprocess
import sys

from setuptools import find_packages, setup

setup(
    name="fdm",
    version="0.2.0",
    author="T. Kehrenberg, M. Bartlett, O. Thomas",
    packages=find_packages(),
    description="Fair distribution matching",
    python_requires=">=3.6",
    package_data={"clustering": ["py.typed"], "fdm": ["py.typed"], "shared": ["py.typed"]},
    install_requires=[
        "EthicML @ git+https://github.com/predictive-analytics-lab/EthicML.git",
        'dataclasses;python_version<"3.7"',
        "gitpython",
        "hydra-core",
        "hydra-ray-launcher",
        "lapjv",
        "numpy >= 1.15",
        "faiss-cpu",
        "pandas >= 0.24",
        "pillow",
        # "pykeops",
        "scikit-image >= 0.14",
        "scikit-learn >= 0.20",
        "scipy >= 1.2.1",
        "torch >= 1.2",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31",
        "typer",
        "wandb >= 0.10.2",
    ],
)
