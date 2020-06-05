#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
etas=( 0.01 0.1 0.5 1.0)

slots=1

for seed in "${seeds[@]}"; do
    for eta in "${etas[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_simple_baselines.py \
        --method dro \
        --eta $eta \
        --filter-labels 2 4 \
        --padding 2 \
        --context-pcnt 0.66666666 \
        --subsample-context 0=0.5 1=1.0 2=0.2 3=0.4 \
        --subsample-train 1=0.3 2=0.0 \
        --scale 0 \
        --gpu 0 \
        --seed $seed \
        --data-split-seed $seed \
        --save-dir experiments/cmnist1missing/baseline/dro/2digits/$seed/0 $@
        sleep 1
    done
done
