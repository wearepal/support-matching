#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 2512 711 142 )

slots=6

echo $scale
for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_both.py @flags/mvp_args.yaml \
    --b-super-val-freq 20 \
    --b-super-val True \
    --b-gpu 0 \
    --c-use-wandb False \
    --b-seed $seed \
    --b-data-split-seed $seed \
    --b-scale 0.0 \
    --d-results cmnist_2digits_$seed\_0.0.csv \
    --b-save-dir experiments/cmnistmissings/ours/full/2digits/$seed/0.0 $@
    sleep 1
done
