#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_no_balancing.py @flags/the_phantom_menace.yaml \
    --b-subsample-train \
    --b-missing-s 0 \
    --b-gpu 0 \
    --d-batch-size 256 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --d-results cmnist_missing_s0_no_cluster_2digits_$seed.csv \
    --b-save-dir experiments/cmnistmissings/ours/full/2digits/$seed $@
    sleep 1
done
