#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
for seed in "${seeds[@]}"; do
    echo $seed
    python run_both.py @flags/adult_pipeline.yaml \
    --b-gpu 0 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-epochs 10 \
    --b-save-dir experiments/test \
    --d-results adult_pipeline.csv "$@"
done