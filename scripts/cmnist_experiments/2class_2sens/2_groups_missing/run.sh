#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

seeds=( 888 1 2410 1996 711 )
etas=( 0.01 0.1 0.5 1.0 )
gpu_id=0
save_dir="experiments/cmnist/2digits/2colors/2missing"

function run_ssl() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_both.py @flags/the_phantom_menace.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed --b-save-dir $save_dir --b-use-wandb False "$@"
    done
}

function run_no_cluster() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_no_balancing.py @flags/the_phantom_menace.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed --b-save-dir $save_dir --b-use-wandb False "$@"
    done
}

function run_baseline() {
    for seed in "${seeds[@]}"; do
        echo $seed
        python run_simple_baselines.py \
        --gpu $gpu_id --seed $seed --data-split-seed $seed --save-dir $save_dir "$@"
    done
}


# UNDERSAMPLE
# ======================== ranking ========================
run_ssl --b-missing-s 0 --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 2group_ranking_undersample.csv "$@"
# ======================== k means ========================
run_ssl --b-missing-s 0 --c-method kmeans --d-results 2group_kmeans_undersample.csv "$@"


# SAMPLING STRATEGY UNUSED
# ===================== no clustering =====================
run_no_cluster  --b-missing-s --d-results 2group_no_clustering.csv "$@"
# ===================== baseline  cnn =====================
run_baseline --dataset cmnist --method cnn --padding 2 --balanced-context False --balanced-test True --biased-train True "$@"
# ===================== baseline  fwd =====================
for eta in "${etas[@]}"; do
    run_baseline --dataset cmnist --method dro --eta $eta --padding 2 --balanced-context False --balanced-test True --biased-train True "$@"
done