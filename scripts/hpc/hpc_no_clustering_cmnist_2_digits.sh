#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_no_balancing.py @flags/the_phantom_menace.yaml \
    --b-gpu 0 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --d-results cmnist_2digits_$seed.csv \
    --b-save-dir experiments/cmnist1missing/ours/full/2digits/$seed $@
    sleep 1
done