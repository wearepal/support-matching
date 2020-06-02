#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 )
etas=( 0.01 0.1 0.5 1.0)

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        for eta in "${etas[@]}"; do
            echo $seed
            python run_simple_baselines.py \
            --method dro \
            --gpu 1 \
            --filter-labels 2 4 \
            --eta $eta \
            --seed $seed \
            --scale $scale \
            --save-dir experiments/cmnist/baseline/dro/2digits/$seed/$eta/$scale $@
            sleep 1
        done
    done
done