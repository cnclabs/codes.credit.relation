#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type='index'
exp_dir=$ROOT/experiments/test_bs/gru12_index

# initialize experiments
init_experiment=false
if $init_experiment; then
    echo initialize experiments
    mkdir -p $exp_dir
    
    params='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines/experiments/params_bs=15786.json' # @cfda4
    cp $params $exp_dir
    mv $exp_dir/params_bs=15786.json $exp_dir/params.json

    for i in {01..13};
    do
        dir=$exp_dir/${type}_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $exp_dir/params.json $exp_dir/${type}_fold_${i}
        echo copy params.json
    done
fi

# run gru ws=12
for i in {01..13};
do
    dir=$exp_dir/${type}_fold_${i}
    python ./PyTorch_Ver/main.py --model_dir=$dir --data_dir=$data_dir/$type/index_fold_${i} --device=$device
done
python ./PyTorch_Ver/pred_folds.py --exp_dir=$exp_dir/ --label_dir=$data_dir/index/ --split_type=index --data_file=test_cum.gz --label_file=test_cum.gz
python ./PyTorch_Ver/synthesize_results.py --parent_dir=$exp_dir
