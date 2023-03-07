#!/bin/bash

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_01/time_fold_${i} --device=1
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_01/ --device 1 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_06/time_fold_${i} --device=1
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_06/ --device 1 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_12/time_fold_${i} --device=1
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_12/ --device 1 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12
