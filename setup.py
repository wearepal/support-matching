from setuptools import find_packages, setup

setup(
    name="suds",
    version="0.2.0",
    author="T. Kehrenberg, M. Bartlett, O. Thomas",
    packages=find_packages(),
    description="Fair distribution matching",
    python_requires=">=3.7",
    package_data={"clustering": ["py.typed"], "suds": ["py.typed"], "shared": ["py.typed"]},
    install_requires=[
        "EthicML == 0.2.0",
        "GitPython >= 2.1.11",
        "gitpython",
        "hydra-core == 1.1.0.dev3",
        "hydra-ray-launcher",
        "lapjv",
        "matplotlib >= 3.0.2, < 3.3.1",
        "numpy >= 1.15",
        "faiss-cpu",
        "pandas >= 1.0",
        "pillow",
        "pipenv >= 2018.11.26",
        "scikit-image >= 0.14",
        "palkit",
        "scikit_learn >= 0.20.1",
        "scipy >= 1.2.1",
        "seaborn >= 0.9.0",
        "teext",
        "torch >= 1.8",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31.1",
        "typer",
        "typing-extensions >= 3.7.2",
        "wandb",
    ],
)
