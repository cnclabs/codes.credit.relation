# /bin/bash

model_dir=$1
data_dir=$2
edge_file_dir=$3
all_company_ids_path=$4
echo model_dir $model_dir
echo data_dir $data_dir
echo edge_file_dir $edge_file_dir
echo all_company_ids_path $all_company_ids_path

device=$5
experiment_type=$6 # index, time, expand
cluster_setting=$7 # industry, kmeas
n_cluster=$8
fold=$9
echo device $device
echo experiment_type $experiment_type
echo cluster_setting $cluster_setting
echo n_cluster $n_cluster
echo fold $fold

## model architecture
lstm_num_units=${10}
intra_gat_hidn_dim=${11}
inter_gat_hidn_dim=${12}
echo lstm_num_units $lstm_num_units
echo intra_gat_hidn_dim $intra_gat_hidn_dim
echo inter_gat_hidn_dim $inter_gat_hidn_dim

## training settings
learning_rate=${13}
weight_decay=${14}
max_epoch=${15}
patience=${16}
gamma=${17}
batch_size=${18}
echo learning_rate $learning_rate
echo weight_decay $weight_decay
echo max_epoch $max_epoch
echo patience $patience
echo gamma $gamma
echo batch_size $batch_size

## experiment settings
model_name=${19}    # NeuDP_GAT
window_size=${20}   # 12
feature_size=${21}  # 14
cum_labels=${22}    # 8
echo model_name $model_name
echo window_size $window_size
echo feature_size $feature_size
echo cum_labels $cum_labels

python3 ../codes/main_valid.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --edge_file_dir $edge_file_dir \
    --n_cluster $n_cluster \
    --cluster_setting $cluster_setting \
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
    --lstm_num_units $lstm_num_units \
    --inter_gat_hidn_dim $inter_gat_hidn_dim \
    --intra_gat_hidn_dim $intra_gat_hidn_dim &&

num_epochs=$(cat $model_dir/num_epochs) &&

python3 ../codes/main.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --edge_file_dir $edge_file_dir \
    --n_cluster $n_cluster \
    --cluster_setting $cluster_setting \
    --device $device \
    --all_company_ids_path $all_company_ids_path \
    --feature_size $feature_size \
    --window_size $window_size \
    --cum_labels $cum_labels \
    --num_epochs $num_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --lstm_num_units $lstm_num_units \
    --inter_gat_hidn_dim $inter_gat_hidn_dim \
    --intra_gat_hidn_dim $intra_gat_hidn_dim &&

python3 ../codes/predict.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --edge_file_dir $edge_file_dir \
    --n_cluster $n_cluster \
    --cluster_setting $cluster_setting \
    --device cpu \
    --all_company_ids_path $all_company_ids_path \
    --feature_size $feature_size \
    --window_size $window_size \
    --cum_labels $cum_labels \
    --data_file test_cum.gz \
    --restore_dir last_weights \
    --lstm_num_units $lstm_num_units \
    --inter_gat_hidn_dim $inter_gat_hidn_dim \
    --intra_gat_hidn_dim $intra_gat_hidn_dim \
    --batch_size $batch_size &&

python3 ../codes/report.py \
    --pred_file $model_dir/${model_name}_pred_test.csv  \
    --label_file $data_dir/test_cum.gz &&

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE