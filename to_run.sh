#!/bin/bash

python run_dis.py -m \
    data=cmnist/3dig \
    bias=cmnist/3dig/4miss \
    enc=mnist \
    fdm=cmnist/fallen_sun \
    misc.log_method=no_cluster \
    fdm.iters=12000 \
    fdm.zs_frac=0.02 \
    fdm.disc_reset_prob=0 \
    fdm.num_disc_updates=2 \
    fdm.num_discs=5 \
    fdm.eval_on_recon=True \
    misc.exp_group=zs2.no_reset.5d.2up.eval_on_recon \
    "$@"
