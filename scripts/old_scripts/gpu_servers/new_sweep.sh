#!/bin/bash

gpu=$1

MAX_SEED=10
seeds=$(seq 1 $MAX_SEED)
flag_file=flags/sleek_sky.yaml
shared_flags="--iters 5000 --elbo-weight 1e2 --warmup 0 --disc-reset-prob 0.0003 --eval-epochs 60"
save_dir="experiments/cmnist/2digits/2colors/1missing/new_toys"

source scripts/gpu_servers/method_loops.sh

if [ "$gpu" -eq "0" ]; then
    echo "gpu=0"
    run_no_cluster --gpu 0 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.031

    run_no_cluster --gpu 0 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.01

    run_no_cluster --gpu 0 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.031 \
        --num-disc-updates 5 --num-discs 1

    run_no_cluster --gpu 0 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 32 --zs-frac 0.031 \
        --num-disc-updates 5 --num-discs 1

    run_no_cluster --gpu 0 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.061
fi

if [ "$gpu" -eq "1" ]; then
    echo "gpu=1"
    run_no_cluster --gpu 1 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 32 --zs-frac 0.031

    run_no_cluster --gpu 1 \
        --aggregator attention --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-1 --enc-init-chans 32 --zs-frac 0.031

    run_no_cluster --gpu 1 \
        --aggregator transposed --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.031

    run_no_cluster --gpu 1 \
        --aggregator transposed --batch-wise-hidden 256 256 --enc-out-dim 128 \
        --kl-weight 1e-2 --enc-init-chans 16 --zs-frac 0.031
fi
