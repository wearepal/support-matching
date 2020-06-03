#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 )

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_no_balancing.py @flags/mvp_args.yaml \
        --b-super-val-freq 20 \
        --b-super-val True \
        --b-gpu 0 \
        --b-batch-size 1000 \
        --b-seed $seed \
        --b-data_split_seed $seed \
        --b-scale $scale \
        --d-results nocluster1k_cmnist_2digits_$seed\_$scale.csv \
        --b-save-dir experiments/cmnist/baseline/nocluster1k/2digits/$seed/$scale $@
        sleep 1
    done
done