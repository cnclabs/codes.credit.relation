#!/bin/bash

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_01/index_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir /tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01_index/ --label_dir /tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_01/ --device 1 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv
python ../pred_folds.py --exp_dir /tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01_index/ --label_dir /tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_01/ --device 1
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm01_index

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_06/index_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_06/ --device 1 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_06/ --device 1
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm06_index

# for i in {01..13};
# do
#     python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_12/index_fold_${i} --device=0
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_12/ --device 1 --all_company_ids_path /tmp2/yhchen/Default_Prediction_Research_Project/data/all_company_ids.csv
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/data/8_labels_index_sample/len_12/ --device 1
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/lstm12_index