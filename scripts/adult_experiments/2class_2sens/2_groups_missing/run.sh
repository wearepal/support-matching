#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

MAX_SEED=100
etas=( 0.01 0.1 0.5 1.0 )
gpu_id=0
save_dir="experiments/adult/2group"

function run_ssl() {
    for seed in $(seq $MAX_SEED); do
        echo $seed
        python run_both.py @flags/adult_pipeline.yaml \
        --a-gpu $gpu_id --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir --a-use-wandb False "$@"
    done
}

function run_no_cluster() {
    for seed in $(seq $MAX_SEED); do
        echo $seed
        python run_no_balancing.py @flags/adult_pipeline.yaml \
        --a-gpu $gpu_id --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir --a-use-wandb False "$@"
    done
}

function run_baseline() {
    for seed in $(seq $MAX_SEED); do
        echo $seed
        python run_simple_baselines.py \
        --gpu $gpu_id --seed $seed --data-split-seed $seed --save-dir $save_dir "$@"
    done
}

# OVERSAMPLE
# ========================== ranking ========================
run_ssl --a-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2group_ranking_oversample.csv --d-oversample True "$@"
# ========================== k means ========================
run_ssl --a-missing-s 0 --c-method kmeans --d-results 2group_kmeans_oversample.csv --d-oversample True "$@"


# UNDERSAMPLE
# ======================== ranking ========================
run_ssl --a-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2group_ranking_undersample.csv --d-oversample False "$@"
# ======================== k means ========================
run_ssl --a-missing-s 0 --c-method kmeans --d-results 2group_kmeans_undersample.csv --d-oversample False "$@"

# TRUE BALANCING
# ===================== no clustering =====================
run_no_cluster --a-missing-s 0 --d-results 2group_true_balance_no_cluster.csv --d-balanced-context True "$@"

# EVAL ON RECON
# ======================== ranking ========================
run_ssl --a-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2group_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ======================== k means ========================
run_ssl --a-missing-s 0 --c-method kmeans --d-results 2group_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"


# SAMPLING STRATEGY UNUSED
# ===================== no clustering =====================
run_no_cluster --a-missing-s 0 --d-results 2group_no_cluster.csv --d-oversample True "$@"
# ===================== baseline  cnn =====================
run_baseline --dataset adult --missing-s 0 --method cnn --padding 2 --balanced-context False --balanced-test True --biased-train True  "$@"
# ===================== baseline  fwd =====================
for eta in "${etas[@]}"; do
    run_baseline --dataset adult --missing-s 0 --method dro --eta $eta --padding 2 --balanced-context False --balanced-test True --biased-train True "$@"
done




