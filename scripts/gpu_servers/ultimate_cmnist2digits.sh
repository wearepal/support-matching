#!/bin/bash

slot=$1

if [ -z "$slot" ]; then
    echo "specify the slot"
    exit 1
fi

flag_file=flags/data_spec/cmnist_2dig_subsampled.yaml
save_dir="experiments/cmnist/2digits/2colors/1missing/ultimate"

if [ "$slot" -eq "1" ]; then
    seeds=$(seq 1 1)
    gpu=0
elif [ "$slot" -eq "2" ]; then
    seeds=$(seq 2 2)
    gpu=0
elif [ "$slot" -eq "3" ]; then
    seeds=$(seq 3 3)
    gpu=1
elif [ "$slot" -eq "4" ]; then
    seeds=$(seq 4 4)
    gpu=1
elif [ "$slot" -eq "5" ]; then
    seeds=$(seq 5 5)
    gpu=2
elif [ "$slot" -eq "6" ]; then
    seeds=$(seq 6 6)
    gpu=2
elif [ "$slot" -eq "7" ]; then
    seeds=$(seq 7 7)
    gpu=3
elif [ "$slot" -eq "8" ]; then
    seeds=$(seq 8 8)
    gpu=3
elif [ "$slot" -eq "9" ]; then
    seeds=$(seq 9 9)
    gpu=0
elif [ "$slot" -eq "10" ]; then
    seeds=$(seq 10 10)
    gpu=1
fi

shared_flags="--gpu $gpu"

echo "seed=$seeds"
echo $shared_flags

source scripts/gpu_servers/method_loops.sh

run_ranking @flags/encoder/mnist.yaml \
    @flags/clustering/vague_spaceship_cluster.yaml @flags/fdm/fallen_sun.yaml

run_ranking_no_predictors @flags/encoder/mnist.yaml \
    @flags/clustering/vague_spaceship_cluster.yaml @flags/fdm/fallen_sun.yaml

run_k_means @flags/encoder/mnist.yaml @flags/fdm/fallen_sun.yaml

run_no_cluster @flags/encoder/mnist.yaml @flags/fdm/fallen_sun.yaml

run_perfect_cluster @flags/encoder/mnist.yaml @flags/fdm/fallen_sun.yaml

run_cnn_baseline

etas=( 0.01 0.1 0.5 1.0 )
run_dro_baseline
