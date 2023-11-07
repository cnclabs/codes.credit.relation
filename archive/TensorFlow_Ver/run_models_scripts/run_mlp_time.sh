#!/bin/bash

for i in {01..13};
do
    python ../train.py --model_dir=../../experiments/mlp01/time_fold_${i} --data_dir=../../data/8_labels_time/len_01/time_fold_${i} --device=0
done
python ../pred_folds.py --exp_dir=../../experiments/mlp01/ --label_dir=../../data/8_labels_time/len_01/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/mlp01

for i in {01..13};
do
    python ../train.py --model_dir=../../experiments/mlp06/time_fold_${i} --data_dir=../../data/8_labels_time/len_06/time_fold_${i} --device=0
done
python ../pred_folds.py --exp_dir=../../experiments/mlp06/ --label_dir=../../data/8_labels_time/len_06/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/mlp06

for i in {01..13};
do
    python ../train.py --model_dir=../../experiments/mlp12/time_fold_${i} --data_dir=../../data/8_labels_time/len_12/time_fold_${i} --device=0
done
python ../pred_folds.py --exp_dir=../../experiments/mlp12/ --label_dir=../../data/8_labels_time/len_12/ --split_type=time
python ../synthesize_results.py --parent_dir=../../experiments/mlp12
