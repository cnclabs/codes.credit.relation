#!/bin/bash

ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data_stratify_13fold'
type='8_labels_index'

# initialize experiments
init_experiment=false
if $init_experiment; then
    echo initialize experiments
    mkdir -p $ROOT/experiments/fim_index
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/fim_index/params.json' # @cfda4
    cp $params $ROOT/experiments/fim_index

    for i in {01..13};
    do
        dir=$ROOT/experiments/fim_index/index_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $ROOT/experiments/fim_index/params.json $ROOT/experiments/fim_index/index_fold_$i
        echo copy params.json
    done
fi

# run fim
for i in {01..13};
do
    dir=$ROOT/experiments/fim_index/index_fold_${i}

    python ./TensorFlow_Ver/train.py --model_dir=$dir --data_dir=$data_dir/$type/len_01_cum_for/index_fold_${i} --device=0
done
python ./TensorFlow_Ver/pred_folds.py --exp_dir=$ROOT/experiments/fim_index/ --label_dir=$data_dir/8_labels_index/len_01_cum_for/ --split_type=index --data_file=test_cum_for.gz --label_file=test_cum_for.gz
python ./TensorFlow_Ver/synthesize_results.py --parent_dir=$ROOT/experiments/fim_index

