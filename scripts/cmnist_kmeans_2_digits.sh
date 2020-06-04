#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
#seeds=( 888 1 2410 1996 511 )
seeds=( 711 142 )

for seed in "${seeds[@]}"; do
    echo $seed
    python run_both.py @flags/mvp_args.yaml \
    --b-super-val-freq 20 \
    --b-super-val True \
    --b-gpu 0 \
    --c-method kmeans \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-scale 0.0 \
    --d-results cmnist_2digits_$seed\_0.0.csv \
    --b-save-dir experiments/cmnist/ours/kmeans/2digits/$seed/0.0 $@
    sleep 1
done
