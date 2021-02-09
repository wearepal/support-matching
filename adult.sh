#!/bin/bash

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=adult/gender \
    bias=adult/partial_outcome \
    enc=adult \
    fdm=adult/simple_on_enc \
    fdm.eval_epochs=60 \
    fdm.balanced_eval=True \
    fdm.iters=6000 \
    fdm.validate=True \
    fdm.aggregator_type=gated \
    fdm.num_disc_updates=3 \
    fdm.pred_s_weight=0.0 \
    fdm.zs_dim=1 \
    fdm.zs_transform=none \
    fdm.batch_size=128 \
    fdm.bag_size=4 \
    fdm.double_adv_loss=False \
    misc.exp_group='agg-${fdm.aggregator_type}.upd-${fdm.num_disc_updates}.preds-${fdm.pred_s_weight}.zst-${fdm.zs_transform}.bs-${fdm.batch_size}.bg-${fdm.bag_size}' \
    misc.seed="range(0,30)" \
    hydra/launcher=ray_0.5gpus \
    "$@"
