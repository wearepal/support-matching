#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 511 )

slots=1

for seed in "${seeds[@]}"; do
    echo $seed
    python run_both.py @flags/pipeline_cmnist_2.yaml \
    --a-val-freq 20 \
    --a-validate True \
    --a-gpu 0 \
    --d-num-disc-updates 3 \
    --d-disc-hidden 256 256 \
    --d-warmup-steps 1000 \
    --d-pred-y-weight 1 \
    --d-train-on-recon False \
    --a-missing-s 0 \
    --a-seed $seed \
    --a-data-split-seed $seed \
    --a-scale 0 \
    --d-results cmnist_2digits_$seed.csv \
    --a-save-dir experiments/cmnistmissings/ours/full/2digits/$seed $@
    sleep 1
done
