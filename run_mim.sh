#!/bin/bash

seeds=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 )

function run_sweep() {
    for seed in "${seeds[@]}"; do
        python run_d.py \
            --balanced-test True \
            --biased-train True \
            --balance-all-quadrants True \
            --dataset adult \
            --drop-native True \
            --input-noise False \
            --mixing-factor 0.0 \
            --batch-size 1000 \
            --disc-hidden-dims 32 \
            --eval-on-recon True \
            --epochs 300 \
            --kl-weight 0 \
            --log-freq 100 \
            --lr 0.001 \
            --num-disc-updates 3 \
            --num-discs 1 \
            --upsample True \
            --train-on-recon True \
            --vae False \
            --val-freq 5 \
            --super-val-freq 10 \
            --super-val True \
            --warmup 200 \
            --weight-decay 0 \
            --zs-frac 0.05714 \
            --init-channels 61 \
            --levels 0 \
            --enc-channels 35 \
            --recon-loss mixed \
            --balanced-context False \
            --data-split-seed $seed \
            --seed $seed \
            "$@"
        sleep 1
    done
}

run_sweep --balanced-context True --result perfect_cluster "$@"
run_sweep --balanced-context False --result no_cluster "$@"
