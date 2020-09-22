#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )

for seed in "${seeds[@]}"; do
    echo $seed
    python run_dis.py @flags/the_phantom_menace.yaml \
    --a-subsample-train \
    --a-missing-s 0 \
    --a-gpu 0 \
    --a-seed $seed \
    --a-data-split-seed $seed \
    --d-results cmnist_2digits_$seed.csv \
    --a-save-dir experiments/cmnist1missing/ours/full/2digits/$seed $@
    sleep 1
done
