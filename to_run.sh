#!/bin/bash

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.longer \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=512 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.big-bag \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.no-disc-reset \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=3 \
    misc.exp_group=tue-sweep.3-discs \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=30.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.less-elbo \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=300.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.more-elbo \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=gated \
    fdm.bag_size=256 \
    fdm.batch_size=1 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.gated \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=cmnist/2dig \
    bias=cmnist/2dig/mildly_subs_miss_s \
    bias.missing_s="[0]" \
    data.context_pcnt=0.5 \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    fdm.aggregator_type=kvq \
    fdm.bag_size=64 \
    fdm.batch_size=8 \
    fdm.disc_reset_prob=0.0003 \
    fdm.disc_weight=0.1 \
    fdm.elbo_weight=100.0 \
    fdm.iters=10_000 \
    fdm.log_freq=450 \
    fdm.num_discs=10 \
    misc.exp_group=tue-sweep.batched \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus
