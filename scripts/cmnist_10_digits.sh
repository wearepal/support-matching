#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 )

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_both.py @flags/cmnist_ae_2_digits.yaml \
        --b-super-val-freq 20 \
        --b-super-val True \
        --b-num-disc-updates 3 \
        --b-disc-hidden 256 256 \
        --b-warmup-steps 1000 \
        --b-seed $seed \
        --b-scale $scale \
        --d-results cmnist_2digits_$seed\_$scale.csv \
        --b-save-dir experiments/cmnist/$seed/$scale $@
        sleep 1
    done
done