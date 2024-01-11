#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type='index'
exp_dir=$ROOT/experiments/fim_index

# initialize experiments
init_experiment=true
if $init_experiment; then
    echo initialize experiments
    mkdir -p $exp_dir
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/fim_index/params.json' # @cfda4
    cp $params $exp_dir

    for i in {01..13};
    do
        dir=$exp_dir/${type}_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $ROOT/experiments/fim_index/params.json $exp_dir/${type}_fold_${i}
        echo copy params.json
    done
fi


# run fim
for i in {01..01};
do
    dir=$exp_dir/${type}_fold_${i}
    python ./TensorFlow_Ver_dev/train_valid.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for/${type}_fold_${i} --device=$device --max_epoch=100 --patience=20
    num_epochs=$(cat ${dir}/num_epochs)
    python ./TensorFlow_Ver_dev/train.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for/${type}_fold_${i} --device=$device --num_epochs=$num_epochs
done
python ./TensorFlow_Ver_dev/pred_folds.py --exp_dir=${exp_dir}/ --label_dir=$data_dir/${type}_01_cum_for/ --split_type=index --data_file=test_cum_for.gz --label_file=test_cum_for.gz
python ./TensorFlow_Ver_dev/synthesize_results.py --parent_dir=${exp_dir}

