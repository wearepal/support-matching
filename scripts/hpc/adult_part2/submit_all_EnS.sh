#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh

seeds=( 888 1 2410 1996 711 )
slots=2
job_file="python.job"
gpu_id=0

function run_sweep() {
    for seed in "${seeds[@]}"; do
        echo $seed
        qsub -pe smpslots $slots $job_file run_both.py @flags/adult_pipeline.yaml \
        --save-dir "../fair-dist-matching/experiments/finn" --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed "$@"
        sleep 1
    done
}

# ========================== ranking ========================
run_sweep --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_EnS.csv "$@"
# ========================== k means ========================
run_sweep --b-missing-s --c-method kmeans --d-results 1group_kmeans_EnS.csv "$@"
