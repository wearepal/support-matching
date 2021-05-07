#!/bin/bash

# perfect cluster
python run_dis.py suds.balanced_context=true misc.log_method=perfect-cluster "$@"

# no cluster
python run_dis.py suds.balanced_context=false misc.log_method=no-cluster-suds "$@"

# ranking
python run_both.py clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-suds "$@"

# kmeans
python run_both.py clust.method=kmeans misc.log_method=kmeans-suds "$@"
