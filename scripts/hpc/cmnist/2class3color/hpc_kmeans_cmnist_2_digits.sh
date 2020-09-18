#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=6

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_both.py @flags/the_phantom_menace.yaml \
    --c-method kmeans \
    --b-gpu 0 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-subsample-train \
    --b-missing-s 2 \
    --b-colors 1 4 8 \
    --b-subsample-context 0=0.5 1=1.0 2=0.3 3=0.2 4=0.4 5=0.2 \
    --d-batch-size 256 \
    --d-results cmnist_kmeans_2digits_3colors.csv \
    --b-save-dir experiments/cmnist/2digits/3colors/missings2/kmeans $@
    sleep 1
done
