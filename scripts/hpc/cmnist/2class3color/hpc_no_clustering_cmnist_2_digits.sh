#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
#    qsub -pe smpslots $slots python-ot.job run_no_balancing.py @flags/the_phantom_menace.yaml \
    python run_no_balancing.py @flags/the_phantom_menace.yaml \
    --b-gpu 0 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-subsample-train \
    --b-missing-s 2 \
    --b-colors 2 4 8 \
    --d-batch-size 256 \
    --d-results cmnist_no_cluster_2digits_3colors.csv \
    --b-save-dir experiments/cmnist/2digits/3colors/missings2/nocluster $@
    sleep 1
done