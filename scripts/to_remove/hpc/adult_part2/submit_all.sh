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
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed "$@"
        sleep 1
    done
}

# supersample

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_supersample.csv --d-upsample True "$@"
# ========================== k means ========================
run_sweep --b-missing-s --c-method kmeans --d-results 1group_kmeans_supersample.csv --d-upsample True "$@"

# ===========================================================
# ====================== 2 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --b-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2groups_ranking_supersample.csv --d-upsample True "$@"
# ========================== k means ========================
run_sweep --b-missing-s 0 --c-method kmeans --d-results 2groups_kmeans_supersample.csv --d-upsample True "$@"

# eval on recon

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ========================== k means ========================
run_sweep --b-missing-s --c-method kmeans --d-results 1group_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"

# ===========================================================
# ====================== 2 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --b-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2groups_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ========================== k means ========================
run_sweep --b-missing-s 0 --c-method kmeans --d-results 2groups_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"
