#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type='expand_len'
exp_dir=$ROOT/experiments/gru12_time_len

# initialize experiments
init_experiment=false
if $init_experiment; then
    echo initialize experiments
    mkdir -p $exp_dir
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/gru12/params.json' # @cfda4
    cp $params $exp_dir

    for i in {01..20};
    do
        dir=$exp_dir/time_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp ${exp_dir}/params.json $exp_dir/time_fold_${i}
        echo copy params.json
    done
fi

# search hyperparameters on fold 1 validation set

# learning_rates=(1e-2 1e-3 1e-4 1e-5)
learning_rate=1e-2
# weight_decays=(1e-3 1e-4 1e-5 1e-6)
weight_decay=1e-6
max_epoch=100
patience=20
gamma=0.9
batch_size=256
lstm_num_units=$3

i=$2

dir=$exp_dir/time_fold_${i}

# make sure data_file is valid_subset_cum.gz
python ./PyTorch_Ver_dev/main_valid.py --model_dir=$dir --data_dir=$data_dir/${type}/time_fold_${i} --label_type cum --device=$device --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma --lstm_num_units=$lstm_num_units &&
num_epochs=$(cat $dir/num_epochs)
python ./PyTorch_Ver_dev/main.py --model_dir=$dir --data_dir=$data_dir/${type}/time_fold_${i} --label_type cum --device=$device --num_epochs=$num_epochs --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma --lstm_num_units=$lstm_num_units &&
# make sure data_file is valid_subset_cum.gz
python ./PyTorch_Ver_dev/predict.py --model_dir=$dir --data_dir=$data_dir/${type}/time_fold_${i} --data_file test_cum.gz --restore_dir last_weights --device cpu &&
# make sure h_params are given, mode as a for append
python ./PyTorch_Ver_dev/report.py --pred_file $dir/pred_test.csv --label_file $data_dir/${type}/time_fold_${i}/test_cum.gz --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma --lstm_num_units=$lstm_num_units