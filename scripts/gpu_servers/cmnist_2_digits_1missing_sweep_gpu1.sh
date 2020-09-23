#!/usr/bin/env bash
#Expects cwd to be fair-dist-matching.

# ===========================================================
# ====================== 1 group missing ====================
# ===========================================================

MAX_SEED=10
seeds=$(seq 1 $MAX_SEED)
flag_file=flags/vague_spaceship_2digits.yaml
shared_flags="--missing-s --d-disc-reset 0.001 --d-iters 3000 --d-disc-weight 0.1 --gpu 1"

# get the experiment loops for the methods
source scripts/gpu_servers/method_loops.sh

save_dir="experiments/cmnist/2digits/2colors/1missing_zs02_lr4"
run_ranking --d-zs-frac 0.02 --d-elbo-weight 100 --d-batch-size 256 --d-warmup 500 --d-lr 1e-4 "$@"

save_dir="experiments/cmnist/2digits/2colors/1missing_bs512"
run_ranking --d-zs-frac 0.1 --d-elbo-weight 100 --d-batch-size 256 --d-warmup 500 "$@"

save_dir="experiments/cmnist/2digits/2colors/1missing_bs512"
run_ranking --d-zs-frac 0.1 --d-elbo-weight 100 --d-batch-size 256 --d-warmup 500 --d-lr 1e-4 "$@"
