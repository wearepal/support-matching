#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
slots=2

for seed in "${seeds[@]}"; do
    echo $seed
    qsub -pe smpslots $slots python-ot.job run_both.py @flags/adult_pipeline.yaml \
    --a-gpu 0 \
    --a-missing-s 0 \
    --c-method pl_enc_no_norm \
    --c-pseudo-labeler ranking \
    --a-seed $seed \
    --a-data-split-seed $seed \
    --d-results 2groups_ranking.csv "$@"
    sleep 1
done
