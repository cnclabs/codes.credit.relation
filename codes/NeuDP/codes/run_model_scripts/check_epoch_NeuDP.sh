# /bin/bash

MODEL_NAME=NeuDP
WINDOW_SIZE=12
FEATURE_SIZE=14
CUM_LABELS=8

ROOT=/home/ybtu/codes.credit.relation.dev/${MODEL_NAME}
# cd $ROOT
DATA_ROOT='/home/cwlin/explainable_credit/data'

all_company_ids_path=$DATA_ROOT/edge_file/all_company_ids.csv

# Fixed parameters
device=$1
experiment_type=$2 # index time
lstm_num_units=$3
learning_rate=$4
weight_decay=$5
max_epoch=100
patience=20
gamma=0.9
batch_size=1


## directory setting
output_file=$ROOT/experiments_alter_epoch/${experiment_type}/${experiment_type}.csv
# output_file=$ROOT/experiments/${experiment_type}/${experiment_type}.csv
echo $output_file

fold_range=(01 02 03 04 05 06 07 08 09 10 11 12 13)
# fold_range=(01)
for fold in "${fold_range[@]}"; do
    
    data_dir=$DATA_ROOT
    if [ "$experiment_type" == "index" ]; then
        data_dir=${data_dir}/index/index_fold_${fold}
    elif [ "$experiment_type" == "time" ]; then
        data_dir=${data_dir}/expand_no_overtime/time_fold_${fold}
    else
        echo "Invalid experiment_type provided!"
        exit 1
    fi

    # setup model directory
    run_id="lstm${lstm_num_units}_lr${learning_rate}_wd${weight_decay}"
    # model_dir=$ROOT/experiments/${experiment_type}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}
    model_dir=$ROOT/experiments_alter_epoch/${experiment_type}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}

    # directory of last_weights has to be made first
    mkdir -p $model_dir/last_weights &&
    echo made $model_dir

    epoch_dir=$ROOT/experiments/${experiment_type}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${experiment_type}_${run_id}
    num_epochs=$(cat $epoch_dir/num_epochs) &&
    if [ "$num_epochs" -lt $max_epoch ]; then
        num_epochs=$((num_epochs - patience))
    fi

    python3 ../codes/main.py --model_dir $model_dir \
    --model_name $MODEL_NAME \
    --data_dir $data_dir \
    --device $device \
    --all_company_ids_path $all_company_ids_path \
    --feature_size $FEATURE_SIZE \
    --window_size $WINDOW_SIZE \
    --cum_labels $CUM_LABELS \
    --num_epochs $num_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --lstm_num_units $lstm_num_units &&

    python3 ../codes/predict.py --model_dir $model_dir \
        --model_name $MODEL_NAME \
        --data_dir $data_dir \
        --device cpu \
        --all_company_ids_path $all_company_ids_path \
        --feature_size $FEATURE_SIZE \
        --window_size $WINDOW_SIZE \
        --cum_labels $CUM_LABELS \
        --data_file test_cum.gz \
        --restore_dir last_weights \
        --batch_size $batch_size \
        --lstm_num_units $lstm_num_units &&

    python3 ../codes/report.py \
        --pred_file $model_dir/${MODEL_NAME}_pred_test.csv  \
        --label_file $data_dir/test_cum.gz &&

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE

    # Extract last 8 lines of AR and RMSNE (adjust the file paths if necessary)
    AR=($(tail -n 8 $model_dir/AR)) &&
    RMSNE=($(tail -n 8 $model_dir/RMSNE)) && 
    num_epochs=$num_epochs &&

    # Compute average AR
    sum_ar=0
    for ar in "${AR[@]}"; do
        sum_ar=$(echo "$sum_ar + $ar" | bc)
    done
    avg_ar=$(echo "scale=4; $sum_ar / ${#AR[@]}" | bc)

    # Compute average RMSNE
    sum_rmsne=0
    for rmsne in "${RMSNE[@]}"; do
        sum_rmsne=$(echo "$sum_rmsne + $rmsne" | bc)
    done
    avg_rmsne=$(echo "scale=4; $sum_rmsne / ${#RMSNE[@]}" | bc)

    # # header: experiment_type,fold,epoch,lstm,lr,wd,AR_01,AR_02,AR_03,AR_04,AR_05,AR_06,AR_07,AR_08,Avg_AR,RMSNE_01,RMSNE_02,RMSNE_03,RMSNE_04,RMSNE_05,RMSNE_06,RMSNE_07,RMSNE_08,Avg_RMSNE
    echo "$experiment_type,$fold,$num_epochs,$lstm_num_units,$learning_rate,$weight_decay,${AR[0]},${AR[1]},${AR[2]},${AR[3]},${AR[4]},${AR[5]},${AR[6]},${AR[7]},$avg_ar,${RMSNE[0]},${RMSNE[1]},${RMSNE[2]},${RMSNE[3]},${RMSNE[4]},${RMSNE[5]},${RMSNE[6]},${RMSNE[7]},$avg_rmsne" >> $output_file
done
