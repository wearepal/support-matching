# this file is meant to be `source`d:
#
#  source scripts/gpu_servers/method_loops.sh
#

# expected environment variables:
#
# seeds
# etas
# flag_file
# shared_flags
# save_dir

function run_ranking() {
    if [ -z "$seeds" ] || [ -z "$flag_file" ] || [ -z "$save_dir" ]; then
        echo "one of 'seeds', 'flag_file', or 'save_dir' is not set"
        exit 1
    fi
    echo "Starting run_ranking"
    for seed in $seeds; do
        echo "seed=$seed"
        cmd="python run_both.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results ranking.csv \
        $shared_flags "$@""
        echo $cmd
        echo ""
        # execute command:
        $cmd
        sleep 5
    done
}

function run_ranking_no_predictors() {
    if [ -z "$seeds" ] || [ -z "$flag_file" ] || [ -z "$save_dir" ]; then
        echo "one of 'seeds', 'flag_file', or 'save_dir' is not set"
        exit 1
    fi
    echo "Starting run_ranking_no_predictors"
    for seed in $seeds; do
        echo "seed=$seed"
        cmd="python run_both.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --c-method pl_enc_no_norm --c-pseudo-labeler ranking --d-results ranking_no_predictors.csv \
        --d-pred-y-weight 0 --d-pred-s-weight 0 \
        $shared_flags "$@""
        echo $cmd
        echo ""
        # execute command:
        $cmd
        sleep 5
    done
}

function run_k_means() {
    echo "Starting run_k_means"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_both.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --c-method kmeans --d-results kmeans.csv \
        $shared_flags "$@"
        sleep 5
    done
}

function run_no_cluster() {
    if [ -z "$seeds" ] || [ -z "$flag_file" ] || [ -z "$save_dir" ]; then
        echo "one of 'seeds', 'flag_file', or 'save_dir' is not set"
        exit 1
    fi
    echo "Starting run_no_cluster"
    for seed in $seeds; do
        echo "seed=$seed"
        cmd="python run_dis.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --d-results no_cluster.csv --d-balanced-context False \
        $shared_flags "$@""
        echo $cmd
        echo ""
        # execute command:
        $cmd
        sleep 5
    done
}

function run_perfect_cluster() {
    if [ -z "$seeds" ] || [ -z "$flag_file" ] || [ -z "$save_dir" ]; then
        echo "one of 'seeds', 'flag_file', or 'save_dir' is not set"
        exit 1
    fi
    echo "Starting run_perfect_cluster"
    for seed in $seeds; do
        echo "seed=$seed"
        cmd="python run_dis.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --d-results perfect_cluster.csv --d-balanced-context True \
        $shared_flags "$@""
        echo $cmd
        echo ""
        # execute command:
        $cmd
        sleep 5
    done
}

function run_cnn_baseline() {
    echo "Starting run_cnn_baseline"
    for seed in $seeds; do
        echo "seed=$seed"
        python run_simple_baselines.py @$flag_file \
        --seed $seed --data-split-seed $seed --save-dir $save_dir \
        --b-method cnn \
        $shared_flags "$@"
        sleep 5
    done
}

function run_dro_baseline() {
    echo "Starting run_dro_baseline"
    for seed in $seeds; do
        echo "seed=$seed"
        for eta in "${etas[@]}"; do
            echo "eta=$eta"
            python run_simple_baselines.py @$flag_file \
            --seed $seed --data-split-seed $seed --save-dir $save_dir \
            --b-method dro --b-eta $eta \
            $shared_flags "$@"
            sleep 5
        done
    done
}
