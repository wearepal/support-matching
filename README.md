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
bash scripts/compare.sh data=adult/gender bias=adult/partial_outcome enc=adult adv=adult/on_enc_with_bags clust=adult
```

For the "no-cluster" baseline, the bag size needs to be changed:

```
python run_dis.py adv.balanced_context=false misc.log_method=no-cluster-adv data=adult/gender bias=adult/partial_outcome enc=adult adv=adult/on_enc_with_bags clust=adult adv.bag_size=32 adv.batch_size=16
```

### Missing subgroup

```
bash scripts/compare.sh data=adult/gender bias=adult/missing_demo enc=adult adv=adult/on_enc_with_bags clust=adult
```

For the "no-cluster" baseline, the bag size needs to be changed:

```
python run_dis.py adv.balanced_context=false misc.log_method=no-cluster-adv data=adult/gender bias=adult/missing_demo enc=adult adv=adult/on_enc_with_bags clust=adult adv.bag_size=32 adv.batch_size=16
```

## Colored MNIST

This dataset will be downloaded automatically.

### 2 digits

#### Subgroup bias

```
bash scripts/compare.sh data=cmnist/2dig bias=cmnist/2dig/subsampled enc=mnist adv=cmnist/simplified clust=vague_spaceship_improved
```

#### Missing subgroup

```
bash scripts/compare.sh data=cmnist/2dig data.context_pcnt=0.5 bias=cmnist/2dig/mildly_subs_miss_s enc=mnist adv=cmnist/gated_3discs clust=vague_spaceship_improved
```

### 3 digits

```
bash scripts/compare.sh data=cmnist/3dig bias=cmnist/3dig/4miss enc=mnist adv=cmnist/mostly_traditional clust=vague_spaceship_improved adv.iters=20000 adv.zs_dim=2
```

## CelebA

The code will try to download this, but the download quota is often saturated,
so it might not work immediately.

```
bash scripts/compare.sh data=celeba/gender_smiling bias=celeba/no_smiling_females enc=mnist adv=cmnist/fallen_sun adv.batch_size=32 adv.bag_size=8 adv.iters=10000 data.context_pcnt=0.5 adv.disc_loss=logistic_ns adv.num_discs=1 adv.disc_reset_prob=0 adv.aggregator_type=gated enc.levels=5 enc.out_dim=128 enc.init_chans=32 adv.enc_loss_w=1 adv.pred_s_loss_w=1
```
