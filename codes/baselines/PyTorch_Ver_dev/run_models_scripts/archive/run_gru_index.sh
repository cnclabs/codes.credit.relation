#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data_stratify_13fold'
type='8_labels_index'

# initialize experiments
init_experiment=true
if $init_experiment; then
    echo initialize experiments
    mkdir -p $ROOT/experiments/gru12_index
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/gru12_index/params.json' # @cfda4
    cp $params $ROOT/experiments/gru12_index

    for i in {01..13};
    do
        dir=$ROOT/experiments/gru12_index/index_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $ROOT/experiments/gru12_index/params.json $ROOT/experiments/gru12_index/index_fold_$i
        echo copy params.json
    done
fi

# run gru ws=12
for i in {01..13};
do
    dir=$ROOT/experiments/gru12_index/index_fold_${i}
    python ./PyTorch_Ver/main.py --model_dir=$dir --data_dir=$data_dir/$type/len_12/index_fold_${i} --device=$device
done
python ./PyTorch_Ver/pred_folds.py --exp_dir=$ROOT/experiments/gru12_index/ --label_dir=$data_dir/8_labels_index/len_12/ --split_type=index --data_file=test_cum.gz --label_file=test_cum.gz
python ./PyTorch_Ver/synthesize_results.py --parent_dir=$ROOT/experiments/gru12_index
