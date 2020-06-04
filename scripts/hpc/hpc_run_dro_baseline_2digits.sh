#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
#seeds=( 888 1 2410 1996 511 )
seeds=( 711 )
etas=( 0.01 0.1 0.5 1.0)

slots=1

for seed in "${seeds[@]}"; do
    for eta in "${etas[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_simple_baselines.py \
        --method dro \
        --missing-s 0 \
        --gpu 0 \
        --filter-labels 2 4 \
        --eta $eta \
        --seed $seed \
        --data-split-seed $seed \
        --scale 0.0 \
        --color-correlation 0.999 \
        --save-dir experiments/cmnist/baseline/dro/2digits/$seed/$eta/0.0 $@
        sleep 1
    done
done
