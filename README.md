# Fair distribution matching

Requires Python 3.6

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

### Partial outcome
```
bash scripts/compare.sh data=adult/gender bias=adult/partial_outcome enc=adult fdm=adult/on_recon_old clust=adult fdm.eval_epochs=60
```

### Missing demographic
```
bash scripts/compare.sh data=adult/gender bias=adult/missing_demo enc=adult fdm=adult/on_recon_old clust=adult fdm.eval_epochs=60
```

## Colored MNIST

This dataset will be downloaded automatically.

### 2 digits

#### Partial outcome
```
bash scripts/compare.sh data=cmnist/2dig bias=cmnist/2dig/mildly_subs enc=mnist fdm=cmnist/fallen_sun clust=vague_spaceship_improved
```

#### Missing demographic
```
bash scripts/compare.sh data=cmnist/2dig bias=cmnist/2dig/mildly_subs_miss_s enc=mnist fdm=cmnist/fallen_sun clust=vague_spaceship_improved
```

### 3 digits
```
bash scripts/compare.sh data=cmnist/3dig bias=cmnist/3dig/4miss enc=mnist fdm=cmnist/fallen_sun clust=vague_spaceship_improved fdm.iters=12000 fdm.zs_frac=0.04
```
