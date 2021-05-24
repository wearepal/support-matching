# Learning with Perfect Bags

Requires Python 3.7.

# Installation

One of the dependencies is PyTorch. If your system is not compatible with the default torch installation
(for example if you require a specific CUDA installation),
then please install this from [pytorch.org](https://pytorch.org/) as required.
If so, it's recommended that you do this first.

We have provided a `setup.py` file with the dependencies.
To install this package, `pip install -e /path/to/this/dir`

# Running Experiments

The `compare.sh` script runs all the variants of our method.

## Adult Income

This dataset is included in the repository.

### Subgroup bias

```
bash scripts/compare.sh data=adult/gender bias=adult/partial_outcome enc=adult adapt=adult/on_enc_with_bags clust=adult
```

For the "no-cluster" baseline, the bag size needs to be changed:

```
python run_ss.py adapt.balanced_context=false misc.log_method=no-cluster-fdm data=adult/gender bias=adult/partial_outcome enc=adult adapt=adult/on_enc_with_bags clust=adult adapt.bag_size=32 adapt.batch_size=16
```

### Missing subgroup

```
bash scripts/compare.sh data=adult/gender bias=adult/missing_demo enc=adult adapt=adult/on_enc_with_bags clust=adult
```

For the "no-cluster" baseline, the bag size needs to be changed:

```
python run_ss.py adapt.balanced_context=false misc.log_method=no-cluster-fdm data=adult/gender bias=adult/missing_demo enc=adult adapt=adult/on_enc_with_bags clust=adult adapt.bag_size=32 adapt.batch_size=16
```

## Colored MNIST

This dataset will be downloaded automatically.

### 2 digits

#### Subgroup bias

```
bash scripts/compare.sh data=cmnist/2dig bias=cmnist/2dig/subsampled enc=mnist adapt=cmnist/simplified clust=vague_spaceship_improved
```

#### Missing subgroup

```
bash scripts/compare.sh data=cmnist/2dig data.context_pcnt=0.5 bias=cmnist/2dig/mildly_subs_miss_s enc=mnist adapt=cmnist/gated_3discs clust=vague_spaceship_improved
```

### 3 digits

```
bash scripts/compare.sh data=cmnist/3dig bias=cmnist/3dig/4miss enc=mnist adapt=cmnist/mostly_traditional clust=vague_spaceship_improved adapt.iters=20000 adapt.zs_dim=2
```

## CelebA

The code will try to download this, but the download quota is often saturated,
so it might not work immediately.

```
bash scripts/compare.sh +experiment=celeba_gender
```
