# this file is meant to be `source`d:
#
#  source scripts/functions.sh
#

function run_ranking() {
    echo "Starting run_ranking"
    for seed in $seeds; do
        echo "seed=$seed"
        cmd="python run_all.py @$flag_file \
        --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir \
        --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results ranking.csv \
        "$shared_flags" "$@" "
        echo $cmd
        echo
        $cmd
        sleep 5
    done
}

function run_k_means() {
    echo "Starting run_k_means"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_all.py @$flag_file \
        --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir \
        --c-method kmeans --d-results kmeans.csv \
        "$shared_flags" "$@"
        sleep 5
    done
}

function run_no_cluster() {
    echo "Starting run_no_cluster"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_no_balancing.py @$flag_file \
        --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir \
        --d-results no_cluster.csv \
        "$shared_flags" "$@"
        sleep 5
    done
}

function run_cnn_baseline() {
    echo "Starting run_cnn_baseline"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_simple_baselines.py @$flag_file \
        --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir \
        --b-method cnn \
        "$shared_flags" "$@"
        sleep 5
    done
}

function run_dro_baseline() {
    echo "Starting run_dro_baseline"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_simple_baselines.py @$flag_file \
        --a-seed $seed --a-data-split-seed $seed --a-save-dir $save_dir \
        --b-method dro \
        "$shared_flags" "$@"
        sleep 5
    done
}
