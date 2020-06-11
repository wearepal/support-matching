#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )

slots=1

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_simple_baselines.py \
    --filter-labels 2 4 \
    --colors 1 4 8 \
    --padding 2 \
    --context-pcnt 0.66666666 \
    --scale 0 \
    --missing-s 2 \
    --method dro \
    --gpu 0 \
    --seed $seed \
    --data-split-seed $seed \
    --save-dir experiments/cmnist/2digits/3colors/baseline/dro $@
    sleep 1
done
