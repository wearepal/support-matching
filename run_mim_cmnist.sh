#!/bin/bash

seeds=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 )

function run_sweep() {
    for seed in "${seeds[@]}"; do
        python run_d.py \
            --background False \
            --binarize True \
            --black True \
            --color-correlation 1.0 \
            --context-pcnt 0.66666666 \
            --data-pcnt 1.0 \
            --dataset cmnist \
            --filter-labels 2 4 \
            --greyscale False \
            --input-noise False \
            --missing-s \
            --padding 2 \
            --quant-level 8 \
            --rotate-data False \
            --scale 0.0 \
            --shift-data False \
            --test-pcnt 0.2 \
            --balanced-context False \
            --batch-size 256 \
            --disc-hidden-dims 256 256 \
            --disc-lr 0.0003 \
            --disc-reset-prob 0.0003 \
            --disc-weight 0.1 \
            --early-stopping 30 \
            --elbo-weight 100.0 \
            --encode-batch-size 1000 \
            --eval-epochs 60 \
            --eval-lr 0.001 \
            --eval-on-recon False \
            --evaluate False \
            --feat-attr False \
            --gamma 1.0 \
            --epochs 300 \
            --kl-weight 0.01 \
            --log-freq 150 \
            --lr 0.0003 \
            --nll-weight 0.01 \
            --num-disc-updates 1 \
            --num-discs 10 \
            --num-workers 4 \
            --upsample True \
            --pred-weight 1.0 \
            --recon-detach True \
            --recon-stability-weight 0 \
            --train-on-recon False \
            --use-inn False \
            --vae False \
            --val-freq 5 \
            --super-val True \
            --super-val-freq 20 \
            --vgg-weight 0 \
            --warmup-steps 0 \
            --weight-decay 0 \
            --zs-frac 0.01 \
            --init-channels 16 \
            --levels 4 \
            --enc-channels 32 \
            --recon-loss l2 \
            --result no_cluster \
            --data-split-seed $seed \
            --seed $seed \
            "$@"
        sleep 1
    done
}

run_sweep --subsample-context 0=0.7 1=0.6 2=0.4 3=1.0 --subsample-train 0=1.0 1=0.4 2=0.0 3=1.0 --save-dir partial "$@"
run_sweep --subsample-context 0=0.7 1=0.6 2=0.4 3=1.0 --subsample-train 0=0.0 1=0.85 2=0.0 3=1.0 --save-dir miss_s "$@"
