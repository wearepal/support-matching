#!/bin/bash

# perfect cluster
python run_dis.py -m fdm.balanced_context=true misc.log_method=perfect-cluster "$@"

# no cluster
python run_dis.py -m fdm.balanced_context=false misc.log_method=no-cluster-fdm "$@"

# ranking
python run_pipeline.py -m clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm "$@"

# kmeans
python run_pipeline.py -m clust.method=kmeans misc.log_method=kmeans-fdm "$@"

# ranking
python run_pipeline.py -m clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm-6 clust.cluster=manual clust.num_clusters=6 "$@"

# kmeans
python run_pipeline.py -m clust.method=kmeans misc.log_method=kmeans-fdm-6 clust.cluster=manual clust.num_clusters=6 "$@"

# ranking
python run_pipeline.py -m clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm-8 clust.cluster=manual clust.num_clusters=8 "$@"

# kmeans
python run_pipeline.py -m clust.method=kmeans misc.log_method=kmeans-fdm-8 clust.cluster=manual clust.num_clusters=8 "$@"
