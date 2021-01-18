#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_dis.py @flags/the_phantom_menace.yaml \
    --a-subsample-train \
    --a-missing-s 0 \
    --d-batch-size 256 \
    --a-subsample-context 0=0.5 1=1.0 2=0.2 3=0.4 \
    --a-gpu 0 \
    --a-seed $seed \
    --a-data-split-seed $seed \
    --d-results cmnist_missing_s0_no_cluster_2digits_$seed.csv \
    --a-save-dir experiments/cmnistmissings/ours/full/2digits/$seed $@
    sleep 1
done
