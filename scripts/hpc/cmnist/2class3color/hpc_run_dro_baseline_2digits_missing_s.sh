#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
etas=( 0.01 0.1 0.5 1.0)

slots=1

for eta in "${etas[@]}"; do
    for seed in "${seeds[@]}"; do
        echo $seed
        echo $eta
        qsub -pe smpslots $slots python-ot.job run_simple_baselines.py \
        --filter-labels 2 4 \
        --colors 1 4 8 \
        --padding 2 \
        --context-pcnt 0.66666666 \
        --scale 0 \
        --missing-s 2 \
        --method dro \
        --eta $eta \
        --gpu 0 \
        --seed $seed \
        --data-split-seed $seed \
        --save-dir experiments/cmnist/2digits/3colors/baseline/dro $@
        sleep 1
    done
done
