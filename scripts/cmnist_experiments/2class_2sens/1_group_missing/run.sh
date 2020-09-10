#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

MAX_SEED=10
seeds=$(seq 1 $MAX_SEED)
etas=( 0.01 0.1 0.5 1.0 )
gpu_id=0
save_dir="experiments/cmnist/2digits/2colors/1missing"

function run_ssl() {
    for seed in $seeds; do
        echo $seed
        python run_both.py @flags/the_phantom_menace.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed --b-save-dir $save_dir --b-use-wandb False "$@"
    done
}

function run_no_cluster() {
    for seed in $seeds; do
        echo $seed
        python run_no_balancing.py @flags/the_phantom_menace.yaml \
        --b-gpu $gpu_id --b-seed $seed --b-data-split-seed $seed --b-save-dir $save_dir --b-use-wandb False "$@"
    done
}

function run_baseline() {
    for seed in $seeds; do
        echo $seed
        python run_simple_baselines.py \
        --gpu $gpu_id --seed $seed --data-split-seed $seed --context-pcnt 0.66666666 --padding 2 --filter-labels 2 4 --scale 0 --balanced-context False --balanced-test True --biased-train True --save-dir $save_dir "$@"
    done
}


# UNDERSAMPLE
# ======================== ranking ========================
run_ssl --b-missing-s --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results 1group_ranking_undersample.csv "$@"
# ======================== k means ========================
run_ssl --b-missing-s --c-method kmeans --d-results 1group_kmeans_undersample.csv "$@"


# SAMPLING STRATEGY UNUSED
# ===================== no clustering =====================
run_no_cluster --b-missing-s --d-results 1group_no_cluster.csv "$@"
# ===================== baseline  cnn =====================
run_baseline --dataset cmnist --method cnn --subsample-train 1=0.3 2=0.0 "$@"
# ===================== baseline  fwd =====================
for eta in "${etas[@]}"; do
    run_baseline --dataset cmnist --method dro --eta $eta --subsample-train 1=0.3 2=0.0 "$@"
done
