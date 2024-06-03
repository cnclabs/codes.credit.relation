# codes.credit.relation

# Inter/intra-Sector

## kmeans/industry clustering

### id mapping issues
- /home/cwlin/explainable_credit/data/prefix_suffix_result.csv

## edge file
```
/home/cwlin/explainable_credit/data/edge_file
.
├── all_company_ids.csv
├── industry_cluster_14
├── industry_cluster_23
├── industry_cluster_62
├── kmeans_cluster_100
├── kmeans_cluster_25
├── kmeans_cluster_50
```


## models 
- ./baselines/
.
├── PyTorch_Ver
│   ├── ar_cap.py
│   ├── main.py
│   ├── model.py
│   ├── pred_folds.py
│   ├── prediction.py
│   ├── predict.py
│   ├── report.py
│   ├── requirements.txt
│   ├── run_models_scripts
│   ├── synthesize_results.py
│   └── utils.py
└── TensorFlow_Ver
    ├── __init__.py
    ├── model
    ├── postprocess
    ├── pred_folds.py
    ├── predict.py
    ├── preprocess
    ├── program_flow
    ├── requirement.txt
    ├── run_models_scripts
    ├── search_hyperparams.py
    ├── synthesize_results.py
    ├── train.py
    └── utils.py

## experiments
experiments template for 1 fold train/test settings
-----

```
# /bin/bash

MODEL_NAME=NeuDP_GAT
EXPERIMENT_TYPE=expand_len

# training settings
WINDOW_SIZE=12
MAX_EPOCH=100
PATIENCE=20
BATCH_SIZE=1
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-06
GAMMA=0.9

# model architecture settings
LSTM_NUM_UNITS=32
CLUSTER_SETTING=industry
N_CLUSTER=14
INTRA_GAT_HIDN_DIM=4
INTER_GAT_HIDN_DIM=4

device=$1
fold=$2
fold=$(printf "%02d" $fold)

run_id="lstm${LSTM_NUM_UNITS}_intra${INTRA_GAT_HIDN_DIM}_inter${INTER_GAT_HIDN_DIM}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"
model_dir=experiments/${MODEL_NAME}/${EXPERIMENT_TYPE}/${CLUSTER_SETTING}_${N_CLUSTER}/fold_${fold}/${MODEL_NAME}_${WINDOW_SIZE}_${EXPERIMENT_TYPE}_${run_id}
data_dir=/home/cwlin/explainable_credit/data/${EXPERIMENT_TYPE}/time_fold_${fold}

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
        --lstm_num_units $LSTM_NUM_UNITS \
        --cluster_setting $CLUSTER_SETTING \
        --n_cluster $N_CLUSTER \
        --intra_gat_hidn_dim $INTRA_GAT_HIDN_DIM \
        --inter_gat_hidn_dim $INTER_GAT_HIDN_DIM &&

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
        --lstm_num_units $LSTM_NUM_UNITS \
        --cluster_setting $CLUSTER_SETTING \
        --n_cluster $N_CLUSTER \
        --intra_gat_hidn_dim $INTRA_GAT_HIDN_DIM \
        --inter_gat_hidn_dim $INTER_GAT_HIDN_DIM &&

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
        --lstm_num_units $LSTM_NUM_UNITS \
        --cluster_setting $CLUSTER_SETTING \
        --n_cluster $N_CLUSTER \
        --intra_gat_hidn_dim $INTRA_GAT_HIDN_DIM \
        --inter_gat_hidn_dim $INTER_GAT_HIDN_DIM &&

python3 report.py \
        --pred_file $model_dir/${MODEL_NAME}_pred_test.csv \
        --label_file $data_dir/test_cum.gz &&

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE  
```