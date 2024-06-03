# /bin/bash

MODEL_NAME=NeuDP
EXPERIMENT_TYPE=expand_len

# training settings
WINDOW_SIZE=12
MAX_EPOCH=100
PATIENCE=20
BATCH_SIZE=1
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-05
GAMMA=0.9

# model architecture settings
LSTM_NUM_UNITS=32

device=$1
fold=$2
fold=$(printf "%02d" $fold)

run_id="lstm${LSTM_NUM_UNITS}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"
model_dir=experiments/${MODEL_NAME}/${EXPERIMENT_TYPE}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${EXPERIMENT_TYPE}_${run_id}
data_dir=../data/${EXPERIMENT_TYPE}/time_fold_${fold}

echo "Model directory: ${model_dir}"


python3 main_valid.py \
        --model_name $MODEL_NAME \
        --device $device \
        --experiment_type $EXPERIMENT_TYPE \
        --fold $fold \
        --max_epoch $MAX_EPOCH \
        --patience $PATIENCE \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --gamma $GAMMA \
        --lstm_num_units $LSTM_NUM_UNITS &&

num_epochs=$(cat $model_dir/num_epochs) &&

python3 main.py \
        --model_name $MODEL_NAME \
        --device $device \
        --experiment_type $EXPERIMENT_TYPE \
        --fold $fold \
        --num_epochs $num_epochs \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --gamma $GAMMA \
        --lstm_num_units $LSTM_NUM_UNITS &&

python3 predict.py \
        --model_name $MODEL_NAME \
        --device $device \
        --experiment_type $EXPERIMENT_TYPE \
        --fold $fold \
        --data_file test_cum.gz \
        --restore_dir last_weights \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --gamma $GAMMA \
        --lstm_num_units $LSTM_NUM_UNITS &&

python3 report.py \
        --pred_file $model_dir/${MODEL_NAME}_pred_test.csv \
        --label_file $data_dir/test_cum.gz &&

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE  