#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching.

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

MAX_SEED=5
seeds=$(seq 1 $MAX_SEED)
etas=( 0.01 0.1 0.5 1.0 )
flag_file=flags/vague_spaceship_2digits.yaml
shared_flags="--missing-s"
save_dir="experiments/cmnist/2digits/2colors/1missing"

# get the experiment loops for the methods
source scripts/gpu_servers/method_loops.sh

# ======================== ranking ========================
run_ranking "$@"
# ======================== k means ========================
run_k_means "$@"

# SAMPLING STRATEGY UNUSED
# ===================== no clustering =====================
run_no_cluster "$@"
# ===================== baseline  cnn =====================
run_cnn_baseline "$@"
# ===================== baseline  fwd =====================
run_dro_baseline "$@"
