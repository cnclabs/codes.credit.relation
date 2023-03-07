#!/bin/bash

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_01/time_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_01/ --device 2 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv --split_type=time
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_01/ --device 2 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_06/time_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_06/ --device 2 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv --split_type=time
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_06/ --device 2 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12/time_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_12/time_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_12/ --device 2 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv --split_type=time
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_time_sample/len_12/ --device 2 --split_type=time
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12
