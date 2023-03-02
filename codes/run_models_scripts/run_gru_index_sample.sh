#!/bin/bash

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments-sample/gru01_index/index_fold_${i} --data_dir=./data/sample/8_labels_index/len_01/index_fold_${i} --num_epochs 300 --device=1 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments-sample/gru01_index/ --label_dir=./data/sample/8_labels_index/len_01/ --device=2 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments-sample/gru01_index

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments-sample/gru06_index/index_fold_${i} --data_dir=./data/sample/8_labels_index/len_06/index_fold_${i} --num_epochs 300 --device=1 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments-sample/gru06_index/ --label_dir=./data/sample/8_labels_index/len_06/ --device=2 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments-sample/gru06_index

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments-sample/gru12_index/index_fold_${i} --data_dir=./data/sample/8_labels_index/len_12/index_fold_${i} --num_epochs 300 --device=1 --batch_size $1
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments-sample/gru12_index/ --label_dir=./data/sample/8_labels_index/len_12/ --device=2 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments-sample/gru12_index

python3 get_results.py --exp_dir /tmp2/cwlin/explainable_credit/explainable_credit/experiments-sample --exp_model gru --exp_date $2 --output_dir /tmp2/cwlin/explainable_credit/results --exp_desc sample.revised.batch_size=$1
