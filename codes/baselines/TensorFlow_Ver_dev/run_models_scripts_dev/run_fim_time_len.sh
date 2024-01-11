#!/bin/bash

device=$1
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/baselines'
cd $ROOT

data_dir='/home/cwlin/explainable_credit/data'
type=expand_len
exp_dir=$ROOT/experiments/fim_time_len

# initialize experiments
init_experiment=false
if $init_experiment; then
    echo initialize experiments
    mkdir -p $exp_dir
    
    params='/tmp2/cywu/default_cumulative/experiments/8_labels/fim_index/params.json' # @cfda4
    cp $params $exp_dir

    for i in {01..20};
    do
        dir=$exp_dir/time_fold_${i}
        mkdir -p $dir/last_weights
        echo made $dir

        cp $ROOT/experiments/fim_index/params.json $exp_dir/time_fold_${i}
        echo copy params.json
    done
fi

# search hyperparameters on fold 1 validation set

# learning_rates=(1e-2 1e-3 1e-4)
learning_rate=1e-3
# weight_decays=(1e-3 1e-4 1e-5)
weight_decay=1e-4
max_epoch=100
patience=20
gamma=0.9
# batch_sizes=(128 256 512)
batch_size=256

i=$2

dir=$exp_dir/time_fold_${i}
python ./TensorFlow_Ver_dev/train_valid.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for_no_overtime/time_fold_${i} --device=$device --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma
num_epochs=$(cat $dir/num_epochs)
python ./TensorFlow_Ver_dev/train.py --model_dir=$dir --data_dir=$data_dir/${type}_01_cum_for_no_overtime/time_fold_${i} --device=$device --num_epochs=$num_epochs --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma
python ./TensorFlow_Ver_dev/predict.py --model_dir $dir --data_dir $data_dir/${type}_01_cum_for_no_overtime/time_fold_${i} --data_file test_cum_for.gz --restore_dir last_weights --device $device
python ./TensorFlow_Ver_dev/postprocess/report.py --pred_file $dir/pred_test.csv --label_file $data_dir/${type}_01_cum_for_no_overtime/time_fold_${i}/test_cum_for.gz --max_epoch=$max_epoch --patience=$patience --learning_rate=$learning_rate --weight_decay=$weight_decay --batch_size=$batch_size --gamma=$gamma