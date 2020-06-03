#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )

slots=1

for seed in "${seeds[@]}"; do
    echo $seed
    python run_both.py @flags/pipeline_cmnist_2.yaml \
    --b-super-val-freq 20 \
    --b-super-val True \
    --b-gpu 0 \
    --d-num-disc-updates 3 \
    --d-disc-hidden 256 256 \
    --d-warmup-steps 1000 \
    --d-pred-weight 1 \
    --d-train-on-recon False \
    --b-missing-s 0 \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-scale 0 \
    --d-results cmnist_2digits_$seed.csv \
    --b-save-dir experiments/cmnistmissings/ours/full/2digits/$seed $@
    sleep 1
done