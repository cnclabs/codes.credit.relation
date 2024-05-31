# /bin/bash

DATA_ROOT='/home/cwlin/explainable_credit/data'

WINDOW_SIZE=12
FEATURE_SIZE=14
CUM_LABELS=8
MODEL_NAME=NeuDP_GAT

# Fixed parameters
device=$1
experiment_type=$2 # index time expand_len expand_time(for inference only)
cluster_setting=$3 # industry 
n_cluster=$4 # 14
lstm_num_units=$5
intra_gat_hidn_dim=$6
inter_gat_hidn_dim=$7
learning_rate=$8
weight_decay=$9
fold_start=${10}
fold_end=${11}
max_epoch=100
patience=20
gamma=0.9
batch_size=1

## directory setting
edge_file_dir=$DATA_ROOT/edge_file 
all_company_ids_path=$DATA_ROOT/edge_file/all_company_ids.csv

output_file=../experiments/${experiment_type}/${experiment_type}.csv
echo $output_file