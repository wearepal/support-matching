#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )

slots=1

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_simple_baselines.py \
    --filter-labels 2 4 \
    --padding 2 \
    --context-pcnt 0.66666666 \
    --subsample-context 0=0.7 1=0.4 2=0.2 3=1.0 \
    --subsample-train 1=0.3 2=0.0 \
    --scale 0 \
    --method cnn \
    --gpu 0 \
    --seed $seed \
    --data-split-seed $seed \
    --save-dir experiments/cmnist1missing/baseline/cnn/2digits/$seed/0 $@
    sleep 1
done
