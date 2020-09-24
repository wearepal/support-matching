#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching.

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

MAX_SEED=10
seeds=$(seq 1 $MAX_SEED)
flag_file=flags/sleek_sky.yaml
shared_flags="--missing-s"
save_dir="experiments/cmnist/2digits/2colors/1missing/no_subsample_lr1e-4"

# get the experiment loops for the methods
source scripts/gpu_servers/method_loops.sh

run_no_cluster --gpu 1 --iters 5000 --lr 1e-4 "$@"
