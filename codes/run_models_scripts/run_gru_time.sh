#!/bin/bash

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru01/time_fold_${i} --data_dir=./data/8_labels_time_no_overtime/len_01/time_fold_${i} --num_epochs 300 --device=0 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru01/ --label_dir=./data/8_labels_time_no_overtime/len_01/ --device=0 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru01

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru06/time_fold_${i} --data_dir=./data/8_labels_time_no_overtime/len_06/time_fold_${i} --num_epochs 300 --device=0 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru06/ --label_dir=./data/8_labels_time_no_overtime/len_06/ --device=0 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru06

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru12/time_fold_${i} --data_dir=./data/8_labels_time_no_overtime/len_12/time_fold_${i} --num_epochs 300 --device=0 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru12/ --label_dir=./data/8_labels_time_no_overtime/len_12/ --device=0 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru12

python3 get_results.py --exp_dir /tmp2/cwlin/explainable_credit/explainable_credit/experiments --exp_model gru --exp_type "time" --exp_date $2 --output_dir /tmp2/cwlin/explainable_credit/results --exp_desc revised.time.batch_size=$1
