# /bin/bash
# set -x

model_dir=$1
data_dir=$2
edge_file_dir=$3 # not used
all_company_ids_path=$4
echo model_dir $model_dir
echo data_dir $data_dir
echo edge_file_dir $edge_file_dir
echo all_company_ids_path $all_company_ids_path

device=$5
experiment_type=$6 # index, time, expand
fold=$7
echo device $device
echo experiment_type $experiment_type
echo cluster_setting $cluster_setting
echo n_cluster $n_cluster
echo fold $fold

## model architecture
lstm_num_units=$8
echo lstm_num_units $lstm_num_units

## training settings
learning_rate=$9
weight_decay=${10}
max_epoch=${11}
patience=${12}
gamma=${13}
batch_size=${14}
echo learning_rate $learning_rate
echo weight_decay $weight_decay
echo max_epoch $max_epoch
echo patience $patience
echo gamma $gamma
echo batch_size $batch_size

## experiment settings
model_name=${15}    # NeuDP_GAT
window_size=${16}          # 12
feature_size=${17}         # 14
cum_labels=${18}            # 8
echo model_name $model_name
echo window_size $window_size
echo feature_size $feature_size
echo cum_labels $cum_labels

python3 ../codes/main_valid.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --device $device \
    --all_company_ids_path $all_company_ids_path \
    --feature_size $feature_size \
    --window_size $window_size \
    --cum_labels $cum_labels \
    --max_epoch $max_epoch \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --lstm_num_units $lstm_num_units &&

num_epochs=$(cat $model_dir/num_epochs) &&

python3 ../codes/predict_valid.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --device cpu \
    --all_company_ids_path $all_company_ids_path \
    --feature_size $feature_size \
    --window_size $window_size \
    --cum_labels $cum_labels \
    --data_file valid_subset_cum.gz \
    --restore_dir last_weights \
    --batch_size $batch_size \
    --lstm_num_units $lstm_num_units &&

python3 ../codes/report.py \
    --pred_file $model_dir/${model_name}_pred_valid.csv  \
    --label_file $data_dir/valid_subset_cum.gz &&

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/valid_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/valid_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE