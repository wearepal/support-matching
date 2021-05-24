#!/bin/bash

# perfect cluster
python run_ss.py adapt.balanced_context=true misc.log_method=perfect-cluster "$@"

# no cluster
python run_ss.py adapt.balanced_context=false misc.log_method=no-cluster-fdm "$@"

# ranking
python run_both_ss.py clust.method=pl_enc_no_norm clust.pseudo_labeler=ranking misc.log_method=ranking-fdm "$@"

# kmeans
python run_both_ss.py clust.method=kmeans misc.log_method=kmeans-fdm "$@"
