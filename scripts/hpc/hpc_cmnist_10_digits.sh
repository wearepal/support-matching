#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 )

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_both.py @flags/pipeline_cmnist_2.yaml \
        --a-enc-out-dim 18 \
        --a-filter-labels \
        --d-zs-frac 0.05555 \
        --a-subsample 0=0.3 1=0.4 2=0.5 3=0.6 4=0.7 5=0.8 6=0.9 \
        --a-super-val-freq 20 \
        --a-super-val True \
        --a-gpu 0 \
        --a-seed $seed \
        --a-data-split-seed $seed \
        --a-scale $scale \
        --d-results cmnist_10digits_$seed\_$scale.csv \
        --a-save-dir experiments/cmnist/ours/full/10digits/$seed/$scale $@
        sleep 1
    done
done
