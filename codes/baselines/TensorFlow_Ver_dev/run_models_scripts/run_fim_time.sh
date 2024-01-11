#!/bin/bash

ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type='expand'
exp_dir=$ROOT/experiments/fim_time_2

# initialize experiments
init_experiment=false
if $init_experiment; then
    echo initialize experiments
    mkdir -p $exp_dir
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/fim/params.json' # @cfda4
    cp $params $exp_dir

    for i in {01..13};
    do
        dir=$exp_dir/${type}_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $ROOT/experiments/fim/params.json $exp_dir/${type}_fold_${i}
        echo copy params.json
    done
fi

# run fim
for i in {01..13};
do
    dir=$exp_dir/time_fold_${i}
    python ./TensorFlow_Ver/train.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for_no_overtime/time_fold_${i} --device=$device
done
python ./TensorFlow_Ver/pred_folds.py --exp_dir=${exp_dir}/ --label_dir=$data_dir/${type}_01_cum_for_no_overtime/ --split_type=time --data_file=test_cum_for.gz --label_file=test_cum_for.gz
python ./TensorFlow_Ver/synthesize_results.py --parent_dir=${exp_dir}
