#!/bin/bash

function run_balanced() {
    # perfect cluster
    python run_dis.py -m fdm.balanced_context=true misc.log_method=perfect-cluster "$@"

    # ranking
    python run_pipeline.py -m clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm "$@"

    # kmeans
    python run_pipeline.py -m clust.method=kmeans misc.log_method=kmeans-fdm "$@"
}

function run_unbalanced() {
    # no cluster
    python run_dis.py -m fdm.balanced_context=false misc.log_method=no-cluster-fdm "$@"
}

run_balanced data=adult/gender bias=adult/partial_outcome enc=adult fdm=adult/on_enc_with_bags_4 clust=adult misc.exp_group=vanilla.bg4 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_balanced data=adult/gender bias=adult/partial_outcome enc=adult fdm=adult/on_enc_with_bags_4 clust=adult fdm.batch_size=64 fdm.bag_size=8 misc.exp_group=bg4-with-8 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_balanced data=adult/gender bias=adult/partial_outcome enc=adult fdm=adult/on_enc_with_bags clust=adult misc.exp_group=vanilla.bg8 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_unbalanced data=adult/gender bias=adult/partial_outcome enc=adult fdm=adult/on_enc_with_bags_4 clust=adult fdm.batch_size=16 fdm.bag_size=32 misc.exp_group=bg4-with-32 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus

run_balanced data=adult/gender bias=adult/missing_demo enc=adult fdm=adult/on_enc_with_bags_4 clust=adult misc.exp_group=vanilla.bg4 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_balanced data=adult/gender bias=adult/missing_demo enc=adult fdm=adult/on_enc_with_bags_4 clust=adult fdm.batch_size=64 fdm.bag_size=8 misc.exp_group=bg4-with-8 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_balanced data=adult/gender bias=adult/missing_demo enc=adult fdm=adult/on_enc_with_bags clust=adult misc.exp_group=vanilla.bg8 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
run_unbalanced data=adult/gender bias=adult/missing_demo enc=adult fdm=adult/on_enc_with_bags_4 clust=adult fdm.batch_size=16 fdm.bag_size=32 misc.exp_group=bg4-with-32 misc.seed="range(0,30)" hydra/launcher=ray_0.5gpus
