#!/bin/bash

results_dir=results/cmnist/2v4/subsampled-0.66cont

function plot() {
    python wandb_csv_to_box_plot.py \
        $results_dir/ranking-suds.simplified.strong_subs.subsampled.csv \
        $results_dir/kmeans-suds.simplified.strong_subs.subsampled.csv \
        $results_dir/no-cluster-suds.simplified.strong_subs.subsampled.csv \
        $results_dir/perfect-cluster.simplified.strong_subs.subsampled.csv \
        $results_dir/cmnist_baseline_cnn_color_60epochs.csv \
        $results_dir/cmnist_baseline_dro_color_eta_0.1_60epochs.csv \
        -f pdf \
        -d 5 2 \
        -c pytorch_classifier -c cnn -c dro \
        -p cmnist_2v4_partial \
        "$@"
}

plot -m acc -x nan 1 "$@"
plot -m ar -x nan 1 "$@"
plot -m tpr -x nan 1 "$@"
plot -m tnr -x nan 1 "$@"
