#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=2

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_both.py @flags/adult_pipeline.yaml \
    --b-gpu 0 \
    --b-missing-s \
    --c-method kmeans \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --d-results 1group_kmeans.csv "$@"
    sleep 1
done
