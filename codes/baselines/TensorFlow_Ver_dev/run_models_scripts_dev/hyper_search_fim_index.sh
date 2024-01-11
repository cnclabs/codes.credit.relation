#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type='index'
exp_dir=$ROOT/experiments/fim_index

# initialize experiments
init_experiment=$2
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

# search hyperparameters on fold 1 validation set
i=01
dir=$exp_dir/${type}_fold_${i}
max_epoch=100
patience=20
batch_sizes=(256)
learning_rates=(1e-2 1e-3 1e-4)
weight_decays=(1e-3 1e-4 1e-5) # 
gamma=0.9

for learning_rate in ${learning_rates[@]};
do
    for weight_decay in ${weight_decays[@]};
    do
        for batch_size in ${batch_sizes[@]};
        do
            # make sure data_file is valid_subset_cum_for.gz
            python ./TensorFlow_Ver_dev/train_valid.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for/${type}_fold_${i} --label_type cum_for --device=$device --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma
            # make sure data_file is valid_subset_cum_for.gz
            python ./TensorFlow_Ver_dev/predict.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for/${type}_fold_${i} --data_file valid_subset_cum_for.gz --restore_dir last_weights --device cpu
            # make sure h_params are given, mode as a for append
            python ./TensorFlow_Ver_dev/postprocess/report.py --pred_file $dir/pred_valid.csv --label_file $data_dir/${type}_01_cum_for/${type}_fold_${i}/valid_subset_cum_for.gz --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma
            echo $learning_rate $weight_decay $batch_size
        done
    done
done

echo done searching hyperparameters

# python3 ./TensorFlow_Ver_dev/postprocess/sort_performances.py --json_file $dir/valid_metrics_pred_best_weights.json --sort_by AR --reverse
python3 ./TensorFlow_Ver_dev/postprocess/sort_performances.py --json_file $dir/valid_metrics_pred_best_weights.json --sort_by RMSNE

echo done sorting performances
