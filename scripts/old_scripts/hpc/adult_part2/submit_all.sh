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
        --a-gpu $gpu_id --a-seed $seed --a-data-split-seed $seed "$@"
        sleep 1
    done
}

# supersample

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --a-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_supersample.csv --d-oversample True "$@"
# ========================== k means ========================
run_sweep --a-missing-s --c-method kmeans --d-results 1group_kmeans_supersample.csv --d-oversample True "$@"

# ===========================================================
# ====================== 2 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --a-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2groups_ranking_supersample.csv --d-oversample True "$@"
# ========================== k means ========================
run_sweep --a-missing-s 0 --c-method kmeans --d-results 2groups_kmeans_supersample.csv --d-oversample True "$@"

# eval on recon

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --a-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ========================== k means ========================
run_sweep --a-missing-s --c-method kmeans --d-results 1group_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"

# ===========================================================
# ====================== 2 group missing ====================
# ===========================================================

# ========================== ranking ========================
run_sweep --a-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2groups_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ========================== k means ========================
run_sweep --a-missing-s 0 --c-method kmeans --d-results 2groups_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"
