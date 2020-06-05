#Expects cwd to be fair-dist-matching. i.e ./scripts/cmnist_1_digits.sh
seeds=( 888 1 2410 1996 711 )
for seed in "${seeds[@]}"; do
    echo $seed
    python run_lr.py \
    --dataset adult \
    --drop-native True \
    --mixing-factor 0.0 \
    --input-noise False \
    --biased-train True \
    --balanced-context False \
    --balanced-test True \
    --data-split-seed $seed \
    --results-csv adult_baseline.csv "$@"
done
