#!/bin/bash

ROOT='/home/ybtu/codes.credit.relation.dev/NeuDP_GAT'
# cd $ROOT
DATA_ROOT='/home/cwlin/explainable_credit/data'

MODEL_NAME=NeuDP_GAT
WINDOW_SIZE=12
FEATURE_SIZE=14
CUM_LABELS=8

# Fixed parameters
device=$1
experiment_type=$2 # index time
cluster_setting=$3 # industry 
n_cluster=$4 # 62
fold=$5 #01
lstm_num_units=$6 

## directory setting
edge_file_dir=$DATA_ROOT/edge_file # 這樣怎麼讀到是哪個cluster的edge? # main_valid裡有寫
all_company_ids_path=$DATA_ROOT/edge_file/all_company_ids.csv

data_dir=$DATA_ROOT
if [ "$experiment_type" == "index" ]; then
    data_dir=${data_dir}/index/index_fold_${fold}
elif [ "$experiment_type" == "time" ]; then
    data_dir=${data_dir}/expand_no_overtime/time_fold_${fold}
else
    echo "Invalid experiment_type provided!"
    exit 1
fi

# Hyperparameter ranges
# lstm_units_range=(128 64 32 16)
# intra_gat_dims_range=(24 16 8 4)
intra_gat_dims_range=(24)
# inter_gat_dims_range=(8 4)
inter_gat_dims_range=(8)
# learning_rates=(0.01 0.001)
learning_rates=(0.001)
# weight_decays=(0.000001 0.00001)
weight_decays=(0.00001)
max_epoch=100
patience=20
gamma=0.9
batch_size=1

output_file=$ROOT/experiments_valid/${experiment_type}/${experiment_type}_valid.csv
echo $output_file

# for lstm_num_units in "${lstm_units_range[@]}"; do
for intra_gat_hidn_dim in "${intra_gat_dims_range[@]}"; do
    for inter_gat_hidn_dim in "${inter_gat_dims_range[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for weight_decay in "${weight_decays[@]}"; do

                # setup model directory
                run_id="lstm${lstm_num_units}_intra${intra_gat_hidn_dim}_inter${inter_gat_hidn_dim}_lr${learning_rate}_wd${weight_decay}"
                model_dir=$ROOT/experiments_valid/${experiment_type}/${cluster_setting}_${n_cluster}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}
                
                # directory of last_weights has to be made first
                mkdir -p $model_dir/last_weights &&
                echo made $model_dir

                bash run_GAT_valid_00.sh $model_dir $data_dir $edge_file_dir $all_company_ids_path \
                                    $device $experiment_type $cluster_setting $n_cluster $fold \
                                    $lstm_num_units $intra_gat_hidn_dim $inter_gat_hidn_dim \
                                    $learning_rate $weight_decay $max_epoch $patience $gamma $batch_size \
                                    $MODEL_NAME $WINDOW_SIZE $FEATURE_SIZE $CUM_LABELS &&
                
                # Extract last 8 lines of AR and RMSNE (adjust the file paths if necessary)
                AR=($(tail -n 8 $model_dir/AR)) &&
                RMSNE=($(tail -n 8 $model_dir/RMSNE)) && 
                num_epochs=$(tail -n 1 $model_dir/num_epochs) &&
                total_params=$(tail -n 1 $model_dir/model_hierarchy.txt | awk '{print $4}') &&

                # header: experiment_type,cluster_setting,n_cluster,fold,model_size,inter,intra,epoch,lstm,lr,wd,AR_01,AR_02,AR_03,AR_04,AR_05,AR_06,AR_07,AR_08,RMSNE_01,RMSNE_02,RMSNE_03,RMSNE_04,RMSNE_05,RMSNE_06,RMSNE_07,RMSNE_08
                echo "$experiment_type,$cluster_setting,$n_cluster,$fold,$total_params,$inter_gat_hidn_dim,$intra_gat_hidn_dim,$num_epochs,$lstm_num_units,$learning_rate,$weight_decay,$batch_size,${AR[0]},${AR[1]},${AR[2]},${AR[3]},${AR[4]},${AR[5]},${AR[6]},${AR[7]},${RMSNE[0]},${RMSNE[1]},${RMSNE[2]},${RMSNE[3]},${RMSNE[4]},${RMSNE[5]},${RMSNE[6]},${RMSNE[7]}" >> $output_file
            done
        done
    done
done
# done
