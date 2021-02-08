#!/bin/bash

python run_dis.py -m \
    fdm.balanced_context=true \
    misc.log_method=perfect-cluster \
    data=adult/gender \
    bias=adult/partial_outcome \
    enc=adult \
    fdm=adult/simple_on_enc \
    clust=adult \
    fdm.eval_epochs=60 \
    fdm.balanced_eval=True \
    fdm.zs_transform=round_ste \
    "$@"
