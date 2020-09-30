#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
etas=( 0.01 0.1 0.5 1.0)

slots=1

for seed in "${seeds[@]}"; do
    for eta in "${etas[@]}"; do
        echo $seed
        python run_simple_baselines.py \
        --dataset adult \
        --method dro \
        --eta $eta \
        --padding 2 \
        --adult-balanced-test True \
        --adult-biased-train True \
        --gpu -1 \
        --seed $seed \
        --data-split-seed $seed $@
    done
done
