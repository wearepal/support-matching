#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_both.py @flags/the_phantom_menace.yaml \
    --c-method kmeans \
    --a-gpu 0 \
    --a-seed $seed \
    --a-data-split-seed $seed \
    --d-results cmnist_1missing_kmeans_2digits_$seed.csv \
    --a-save-dir experiments/cmnist1missing/baseline/kmeans/2digits/$seed $@
    sleep 1
done
