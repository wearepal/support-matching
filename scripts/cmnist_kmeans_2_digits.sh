#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 )

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_both.py @flags/pipeline_cmnist_2.yaml \
        --b-super-val-freq 20 \
        --b-super-val True \
        --b-gpu 0 \
        --c-use-wandb False \
        --c-method kmeans \
        --b-seed $seed \
        --b-data-split-seed $seed \
        --b-scale $scale \
        --d-results cmnist_2digits_$seed\_$scale.csv \
        --b-save-dir experiments/cmnist/ours/kmeans/2digits/$seed/$scale $@
        sleep 1
    done
done