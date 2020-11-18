#!/bin/bash

# ranking
python run_pipeline.py -m clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm "$@"

# kmeans
python run_pipeline.py -m clust.method=kmeans misc.log_method=kmeans-fdm "$@"

# no cluster
python run_dis.py -m fdm.balanced_context=false misc.log_method=no-cluster-fdm "$@"

# perfect cluster
python run_dis.py -m fdm.balanced_context=true misc.log_method=perfect-cluster "$@"
