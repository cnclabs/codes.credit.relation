# !/bin/bash
model_path="/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments"

# all or sample
sample_dataset_path="/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/"
all_dataset_path="/tmp2/yhchen/Default_Prediction_Research_Project/data/"

dataset_path=$sample_dataset_path

# cross-time or cross-section
# dataset_type="time"
# model_path_suffix=
# dataset_path_suffix="_no_overtime"
dataset_type="index"
model_path_suffix="_index"
dataset_path_suffix=

# Model Setting
# batch_norm=on
# lstm_num_units=64
# gru_model=GRU
number_of_layer=2

# Hyperparameter Setting
# learning_rate=1e-4
# batch_size=256
# num_epochs=20
# dropout_rate=0.5
# weight_decay=1e-5
# patience=20
# num_epochs=300
# alpha=0.2


for window_size in 01 06 12
do
    for i in {01..13};
    do
        python ../main.py \
        --model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i} \
        --data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_${i} \
        --device=0 \
        --gru_model=Graph_GRU \
        --number_of_layer=$number_of_layer
    done

    python ../pred_folds.py \
    --exp_dir=${model_path}/gru${window_size}${model_path_suffix}/ \
    --label_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/ \
    --device=cpu

    python ../synthesize_results.py \
    --parent_dir=${model_path}/gru${window_size}${model_path_suffix}
done

python ../get_results.py \
--exp_dir=${model_path} \
--exp_model=gru \
--exp_type=$dataset_type \
--exp_date=$(date +%m%d) \
--exp_desc=GRU_Shared_all_time_wo_relation \
--output_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/results


######################################################################################################
# for i in {01..13};
# do
#     python ../main.py \
#     --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index/index_fold_${i} \
#     --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_01/index_fold_${i} \
#     --device=0 \
#     --gru_model=Graph_GRU_shared
#     # --adgat_relation
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_01/ --device cpu
# python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru01_index

# for i in {01..13};
# do
#     python ../main.py \
#     --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index/index_fold_${i} \
#     --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_06/index_fold_${i} \
#     --device=0 \
#     --gru_model=Graph_GRU_shared
#     # --adgat_relation
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_06/ --device cpu
# python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru06_index

# for i in {01..13};
# do
#     python ../main.py \
#     --model_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index/index_fold_${i} \
#     --data_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_12/index_fold_${i} \
#     --device=0 \
#     --gru_model=Graph_GRU_shared
#     # --adgat_relation
# done
# python ../pred_folds.py --exp_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index/ --label_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_index/len_12/ --device cpu
# python ../synthesize_results.py --parent_dir=/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments/gru12_index
