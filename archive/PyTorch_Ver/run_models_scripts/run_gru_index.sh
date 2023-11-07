#!/bin/bash

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru01_index/index_fold_${i} --data_dir=../../data/8_labels_index/len_01/index_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru01_index/ --label_dir=../../data/8_labels_index/len_01/
python ../synthesize_results.py --parent_dir=../../experiments/gru01_index

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru06_index/index_fold_${i} --data_dir=../../data/8_labels_index/len_06/index_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru06_index/ --label_dir=../../data/8_labels_index/len_06/
python ../synthesize_results.py --parent_dir=../../experiments/gru06_index

for i in {01..13};
do
    python ../main.py --model_dir=../../experiments/gru12_index/index_fold_${i} --data_dir=../../data/8_labels_index/len_12/index_fold_${i} --device=cpu
done
python ../pred_folds.py --exp_dir=../../experiments/gru12_index/ --label_dir=../../data/8_labels_index/len_12/
python ../synthesize_results.py --parent_dir=../../experiments/gru12_index
