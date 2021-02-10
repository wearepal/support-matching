#!/bin/bash

# python run_dis.py -m \
#     fdm.balanced_context=true \
#     misc.log_method=perfect-cluster \
python run_dis.py -m \
    fdm.balanced_context=false \
    misc.log_method=no-cluster-fdm \
    data=adult/gender \
    bias=adult/partial_outcome \
    enc=adult \
    fdm=adult/simple_on_enc \
    fdm.eval_epochs=60 \
    fdm.balanced_eval=True \
    fdm.iters=6000 \
    fdm.validate=False \
    fdm.aggregator_type=kvq \
    fdm.num_disc_updates=3 \
    fdm.pred_s_weight=0.0,1.0 \
    fdm.pred_y_weight=0.0,1.0 \
    fdm.zs_dim=1 \
    fdm.zs_transform=none,round_ste \
    fdm.batch_size=64 \
    fdm.bag_size=8 \
    fdm.double_adv_loss=False \
    misc.exp_group='preds-${fdm.pred_s_weight}.predy-${fdm.pred_y_weight}.zst-${fdm.zs_transform}' \
    misc.seed="range(0,20)" \
    hydra/launcher=ray_0.33gpus \
    "$@"
