# !/bin/bash

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_01/index_fold_${i} --device=2
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_01/ --device cpu
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_06/index_fold_${i} --device=2
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_06/ --device cpu
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index

for i in {01..13};
do
    python ../main.py --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index/index_fold_${i} --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_12/index_fold_${i} --device=2
done
python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_12/ --device cpu
python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index
