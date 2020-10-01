#!/bin/bash

data_flags=flags/data_spec/cmnist_2dig_mildly_subsampled.yaml
save_dir=experiments/cmnist/2digits/2colors/1missing/mildly_subsampled
encoder_flags=flags/encoder/mnist.yaml
fdm_flags=flags/fdm/fallen_sun.yaml
clust_flags=flags/clustering/vague_spaceship_cluster.yaml

slot=$1

if [ -z "$slot" ]; then
    echo "please specify a slot"
    exit 1
fi

if [ "$slot" -eq "0" ]; then
    seeds=$(seq 10 19)
    gpu=0
elif [ "$slot" -eq "1" ]; then
    seeds=$(seq 20 29)
    gpu=1
fi

# =================== set parameters for sweep and import functions ===================

shared_flags="--gpu $gpu"
flag_file=$data_flags

echo "seed=$seeds"
echo $shared_flags

source scripts/gpu_servers/method_loops.sh

# ============== run all methods ================

run_ranking @$encoder_flags @$clust_flags @$fdm_flags

run_ranking_no_predictors @$encoder_flags @$clust_flags @$fdm_flags \
    --d-pred-y-weight 0 --d-pred-s-weight 0

run_k_means @$encoder_flags @$fdm_flags

run_no_cluster @$encoder_flags @$fdm_flags

run_perfect_cluster @$encoder_flags @$fdm_flags

run_cnn_baseline

etas=( 0.01 0.1 0.5 1.0 )
run_dro_baseline
