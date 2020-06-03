#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )
scales=( 0.0 )

slots=1

for scale in "${scales[@]}"; do
    echo $scale
    for seed in "${seeds[@]}"; do
        echo $seed
        qsub -pe smpslots $slots python-ot.job run_no_balancing.py @flags/pipeline_cmnist_2.yaml \
        --b-super-val-freq 20 \
        --b-super-val True \
        --b-gpu 0 \
        --b-seed $seed \
        --b-data-split-seed $seed \
        --b-scale $scale \
        --d-results nocluster_cmnist_2digits_$seed\_$scale.csv \
        --b-save-dir experiments/cmnist/baseline/nocluster/2digits/$seed/$scale $@
        sleep 1
    done
done