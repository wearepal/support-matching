# Addressing Attribute Bias with Adversarial Support-Matching

Code for the paper [Addressing Attribute Bias with Adversarial Support-Matching](https://openreview.net/forum?id=JYbnJ92TJf).

Requires Python 3.10+.

# Installation

One of the dependencies is PyTorch. If your system is not compatible with the default torch installation
(for example if you require a specific CUDA installation),
then please install this from [pytorch.org](https://pytorch.org/) as required.
If so, it's recommended that you do this first.

You can install this package, with dependencies, with `pip install .`.

## Running Experiments

### ACS

This dataset will be downloaded on first use.

```
python -m src.run.supmatch +experiment=acs/fcn
```

### NICO++

This dataset has to be downloaded separately.

```
python -m src.run.supmatch +experiment=nicopp/rn50/pretrained_enc ds.root=/path/to/dataset
```

### Colored MNIST

This dataset will be downloaded automatically.

```
python -m src.run.supmatch +experiment=cmnist/2d2c labeller=gt
```

### CelebA

The code will try to download this, but the download quota is often saturated,
so it might not work immediately.

#### Without smiling males
```
python -m src.run.supmatch +experiment=celeba/sm/pt split=celeba/artifact/no_smiling_males
```

#### Without smiling females
```
python -m src.run.supmatch +experiment=celeba/sm/pt split=celeba/artifact/no_smiling_females
```

#### Without unsmiling males
```
python -m src.run.supmatch +experiment=celeba/sm/pt split=celeba/artifact/no_unsmiling_males
```

#### Without unsmiling females
```
python -m src.run.supmatch +experiment=celeba/sm/pt split=celeba/artifact/no_unsmiling_females
```
