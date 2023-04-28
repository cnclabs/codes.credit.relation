# !/bin/bash

################  Dataset Setting  ###################
# all or sample
sample_company=500
# sample_dataset_path="/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/"
# all_dataset_path="/tmp2/yhchen/Default_Prediction_Research_Project/data/"
sample_dataset_path="../data/sample${sample_company}/"
all_dataset_path="../data/all/"

dataset_path=$sample_dataset_path

# cross-time
dataset_type="time"
model_path_suffix=
dataset_path_suffix="_no_overtime"
#######################################################

#################  Model Setting  #####################
# model_path="/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments"
model_path="../../experiments"

# lstm_num_units=64
# gru_model=GRU
# number_of_layer=2
# adgat_relation
#######################################################

#############  Hyperparameter Setting  #################
# learning_rate=1e-4
# batch_size=1
# num_epochs=20
# dropout_rate=0.5
# weight_decay=1e-5
# patience=20
# num_epochs=300
# alpha=0.2
#######################################################

# Clean last experiments data
for window_size in 01 06 12
do
    # clean num_epochs
    if [ -f "${model_path}/gru${window_size}${model_path_suffix}/num_epochs" ]
    then
        rm ${model_path}/gru${window_size}${model_path_suffix}/num_epochs
    fi

    # clean results_pt.csv
    if [ -f "${model_path}/gru${window_size}${model_path_suffix}/results_pt.csv" ]
    then
        rm ${model_path}/gru${window_size}${model_path_suffix}/results_pt.csv
    fi

    for i in {01..13};
    do
        # clean checkpoint in last weight
        if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/last_weights/checkpoint" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/last_weights/checkpoint
        fi
        # clean fixed_pred.csv
            if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/fixed_pred.csv" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/fixed_pred.csv
        fi
        # clean metrics_eval_best_weights.json
        if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/metrics_eval_best_weights.json" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/metrics_eval_best_weights.json
        fi
        # clean metrics_eval_last_weights.json
        if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/metrics_eval_last_weights.json" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/metrics_eval_last_weights.json
        fi
        # clean metrics_eval_last_weights.json
        if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/pred.csv" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/pred.csv
        fi
        # clean metrics_eval_last_weights.json
        if [ -f "${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/test_metrics_pred_best_weights.json" ]
        then
            rm ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/test_metrics_pred_best_weights.json
        fi
        # clean train.log
        > ${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i}/train.log
    done
done

# Start Experiments
device=0

# The parameter setting when window size equal to 1
window_size=01
batch_size=1

# Decide the number of epoch by evaluating the first folder of dataset (need to give get_epoch parameter)
python ../main.py \
--model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_01 \
--data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_01 \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--shared_parameter \
--adgat_relation \
--get_epoch \
--batch_size=${batch_size} \
--epoch_dir=${model_path}/gru${window_size}${model_path_suffix}

for i in {01..13};
do
    python ../main.py \
    --model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i} \
    --data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_${i} \
    --all_company_ids_path=${dataset_path}all_company_ids.csv \
    --device=${device} \
    --shared_parameter \
    --adgat_relation \
    --batch_size=${batch_size} \
    --epoch_dir=${model_path}/gru${window_size}${model_path_suffix}
done

python ../pred_folds.py \
--exp_dir=${model_path}/gru${window_size}${model_path_suffix}/ \
--label_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/ \
--split_type=${dataset_type} \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--batch_size=${batch_size}

python ../synthesize_results.py \
--parent_dir=${model_path}/gru${window_size}${model_path_suffix}

# The parameter setting when window size equal to 6
window_size=06
batch_size=5

# Decide the number of epoch by evaluating the first folder of dataset (need to give get_epoch parameter)
python ../main.py \
--model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_01 \
--data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_01 \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--shared_parameter \
--adgat_relation \
--get_epoch \
--batch_size=${batch_size} \
--epoch_dir=${model_path}/gru${window_size}${model_path_suffix}

for i in {01..13};
do
    python ../main.py \
    --model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i} \
    --data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_${i} \
    --all_company_ids_path=${dataset_path}all_company_ids.csv \
    --device=${device} \
    --shared_parameter \
    --adgat_relation \
    --batch_size=${batch_size} \
    --epoch_dir=${model_path}/gru${window_size}${model_path_suffix}
done

python ../pred_folds.py \
--exp_dir=${model_path}/gru${window_size}${model_path_suffix}/ \
--label_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/ \
--split_type=${dataset_type} \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--batch_size=${batch_size}

python ../synthesize_results.py \
--parent_dir=${model_path}/gru${window_size}${model_path_suffix}

# The parameter setting when window size equal to 12
window_size=12
batch_size=10

# Decide the number of epoch by evaluating the first folder of dataset (need to give get_epoch parameter)
python ../main.py \
--model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_01 \
--data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_01 \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--shared_parameter \
--adgat_relation \
--get_epoch \
--batch_size=${batch_size} \
--epoch_dir=${model_path}/gru${window_size}${model_path_suffix}

for i in {01..13};
do
    python ../main.py \
    --model_dir=${model_path}/gru${window_size}${model_path_suffix}/${dataset_type}_fold_${i} \
    --data_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/${dataset_type}_fold_${i} \
    --all_company_ids_path=${dataset_path}all_company_ids.csv \
    --device=${device} \
    --shared_parameter \
    --adgat_relation \
    --batch_size=${batch_size} \
    --epoch_dir=${model_path}/gru${window_size}${model_path_suffix}
done

python ../pred_folds.py \
--exp_dir=${model_path}/gru${window_size}${model_path_suffix}/ \
--label_dir=${dataset_path}8_labels_${dataset_type}${dataset_path_suffix}/len_${window_size}/ \
--split_type=${dataset_type} \
--all_company_ids_path=${dataset_path}all_company_ids.csv \
--device=${device} \
--batch_size=${batch_size}

python ../synthesize_results.py \
--parent_dir=${model_path}/gru${window_size}${model_path_suffix}

python ../get_results.py \
--exp_dir=${model_path} \
--exp_model=gru \
--exp_type=${dataset_type} \
--exp_date=$(date +%m%d) \
--exp_desc=test \
--output_dir=../../results/
