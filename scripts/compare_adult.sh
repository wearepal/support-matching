#!/bin/bash

data=adult/gender
enc=adult
clust=adult
suds=adult/base

# ranking
python run_both.py \
    data=$data \
    enc=$enc \
    clust=$clust \
    suds=$suds \
    clust.method=pl_enc_no_norm \
    clust.pseudo_labeler=ranking \
    misc.log_method=ranking-suds \
    "$@"

# kmeans
python run_both.py \
    data=$data \
    enc=$enc \
    clust=$clust \
    suds=$suds \
    clust.method=kmeans \
    misc.log_method=kmeans-suds \
    "$@"

# no cluster
python run_dis.py \
    data=$data \
    enc=$enc \
    suds=$suds \
    suds.balanced_context=false \
    misc.log_method=no-cluster-suds \
    "$@"

# perfect cluster
python run_dis.py \
    data=$data \
    enc=$enc \
    suds=$suds \
    suds.balanced_context=true \
    misc.log_method=perfect-cluster \
    "$@"
