#!/bin/bash

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru01/time_fold_${i} --data_dir=../../data/8_labels_time/len_01/time_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru01/ --label_dir=../../data/8_labels_time/len_01/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/gru01

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru06/time_fold_${i} --data_dir=../../data/8_labels_time/len_06/time_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru06/ --label_dir=../../data/8_labels_time/len_06/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/gru06

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru12/time_fold_${i} --data_dir=../../data/8_labels_time/len_12/time_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru12/ --label_dir=../../data/8_labels_time/len_12/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/gru12
