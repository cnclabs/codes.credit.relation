# /bin/bash

# TODO: make the ROOT to relative path
ROOT='/tmp2/ybtu/codes.credit.relation/codes'
# cd $ROOT
DATA_ROOT='/home/cwlin/explainable_credit/data'

# MODEL_NAME=NeuDP_GAT_wo_intra # NeuDP_GAT, NeuDP_GAT_wo_intra, NeuDP_GAT_wo_inter
# MODEL_NAME=NeuDP_GAT_wo_inter

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
MODEL_NAME=${10} # NeuDP_GAT, NeuDP_GAT_wo_intra, NeuDP_GAT_wo_inter
# fold_start=${10}
# fold_end=${11}
max_epoch=100
patience=20
gamma=0.9
batch_size=1

## directory setting
edge_file_dir=$DATA_ROOT/edge_file 
all_company_ids_path=$DATA_ROOT/edge_file/all_company_ids.csv

output_file=$ROOT/experiments/${experiment_type}/${experiment_type}.csv
echo $output_file

fold_range=(01)
# for fold in "${fold_range[@]}"; do
#     fold=$(printf "%02d" $fold)

#     data_dir=$DATA_ROOT
#     if [ "$experiment_type" == "index" ]; then
#         data_dir=${data_dir}/index/index_fold_${fold}
#     elif [ "$experiment_type" == "time" ]; then
#         data_dir=${data_dir}/expand_no_overtime/time_fold_${fold}
#     elif [ "$experiment_type" == "expand_len" ]; then
#         data_dir=${data_dir}/expand_len/time_fold_${fold}
#     else
#         echo "Invalid experiment_type provided!"
#         exit 1
#     fi

#     # setup model directory
#     run_id="lstm${lstm_num_units}_intra${intra_gat_hidn_dim}_inter${inter_gat_hidn_dim}_lr${learning_rate}_wd${weight_decay}"
#     model_dir=$ROOT/experiments/${experiment_type}/${cluster_setting}_${n_cluster}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}

#     # directory of last_weights has to be made first
#     mkdir -p $model_dir/last_weights &&
#     echo made $model_dir
# done

# # for fold in $(seq $fold_end -1 $fold_start); do
for fold in "${fold_range[@]}"; do
    fold=$(printf "%02d" $fold)

    data_dir=$DATA_ROOT
    if [ "$experiment_type" == "index" ]; then
        data_dir=${data_dir}/index/index_fold_${fold}
    elif [ "$experiment_type" == "time" ]; then
        data_dir=${data_dir}/expand_no_overtime/time_fold_${fold}
    elif [ "$experiment_type" == "expand_len" ]; then
        data_dir=${data_dir}/expand_len/time_fold_${fold}
    else
        echo "Invalid experiment_type provided!"
        exit 1
    fi

    # setup model directory
    run_id="lstm${lstm_num_units}_intra${intra_gat_hidn_dim}_inter${inter_gat_hidn_dim}_lr${learning_rate}_wd${weight_decay}"
    model_dir=$ROOT/experiments/${experiment_type}/${cluster_setting}_${n_cluster}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}

    if [ -f "$model_dir/AR" ] && [ -f "$model_dir/RMSNE" ] && [ -f "$model_dir/num_epochs" ]; then
        echo "AR, RMSNE, and num_epochs already exist. Skipping experiment."
    else
        bash run_GAT.sh $model_dir $data_dir $edge_file_dir $all_company_ids_path \
                        $device $experiment_type $cluster_setting $n_cluster $fold \
                        $lstm_num_units $intra_gat_hidn_dim $inter_gat_hidn_dim \
                        $learning_rate $weight_decay $max_epoch $patience $gamma $batch_size \
                        $MODEL_NAME $WINDOW_SIZE $FEATURE_SIZE $CUM_LABELS
    fi
done
