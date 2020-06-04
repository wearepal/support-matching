#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )

slots=1

for seed in "${seeds[@]}"; do
    echo $seed
    python run_simple_baselines.py \
    --missing-s 0 \
    --method cnn \
    --gpu 1 \
    --filter-labels 2 4 \
    --seed $seed \
    --data-split-seed $seed \
    --scale 0.0 \
    --color-correlation 0.999 \
    --save-dir experiments/cmnistmissings/baseline/cnn/2digits/$seed/0 $@
    sleep 1
done
