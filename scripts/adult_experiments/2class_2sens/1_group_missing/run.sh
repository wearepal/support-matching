#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

seeds=( 888 1 2410 1996 711 )
etas=( 0.01 0.1 0.5 1.0 )
gpu_id=0

function run_ssl() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_both.py @flags/adult_pipeline.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed "$@"
        sleep 1
    done
}

function run_no_cluster() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_no_clustering.py @flags/adult_pipeline.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed "$@"
        sleep 1
    done
}

function run_baseline() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_simple_baselines.py \
        --gpu -1 --seed $seed --data-split-seed $seed "$@"
        sleep 1
    done
}

# SUPERSAMPLE
# ========================== ranking ========================
run_ssl --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_supersample.csv --d-upsample True "$@"
# ========================== k means ========================
run_ssl --b-missing-s --c-method kmeans --d-results 1group_kmeans_supersample.csv --d-upsample True "$@"


# SUBSAMPLE
# ======================== ranking ========================
run_ssl --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_subsample.csv "$@"
# ======================== k means ========================
run_ssl --b-missing-s --c-method kmeans --d-results 1group_kmeans_subsample.csv "$@"

# EVAL ON RECON
# ======================== ranking ========================
run_ssl --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_eval_on_recon.csv --d-eval-on-recon True "$@"
# ======================== k means ========================
run_ssl --b-missing-s --c-method kmeans --d-results 1group_kmeans_eval_on_recon.csv --d-eval-on-recon True "$@"


# SAMPLING STRATEGY UNUSED
# ===================== no clustering =====================
run_no_cluster  --b-missing-s --d-results 1group_ranking_supersample.csv --d-upsample True "$@"
# ===================== baseline  cnn =====================
run_baseline --dataset adult --method cnn --padding 2 --balanced-context False --balanced-test True --biased-train True "$@"
# ===================== baseline  fwd =====================
for eta in "${etas[@]}"; do
    run_baseline --dataset adult --method dro --eta $eta --padding 2 --balanced-context False --balanced-test True --biased-train True "$@"
done




