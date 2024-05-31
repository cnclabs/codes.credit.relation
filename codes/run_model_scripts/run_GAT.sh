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
echo model_name $model_name

python3 ../main_valid.py --model_dir $model_dir \
    --model_name $model_name \
    --data_dir $data_dir \
    --edge_file_dir $edge_file_dir \
    --n_cluster $n_cluster \
    --cluster_setting $cluster_setting \
    --device $device \
    --all_company_ids_path $all_company_ids_path \
    --max_epoch $max_epoch \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --lstm_num_units $lstm_num_units \
    --inter_gat_hidn_dim $inter_gat_hidn_dim \
    --intra_gat_hidn_dim $intra_gat_hidn_dim &&

num_epochs=$(cat $model_dir/num_epochs)