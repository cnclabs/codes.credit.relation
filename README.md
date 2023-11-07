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

# TODO
- experiments
1. dataset original (stratefy kfold. k=13)
    * cross section
    * FIM
    * NeuDP batch=256
    * NeuDP all company
    * NeuDP_GAT
        * kmeans
        * industry
    * cross time
    * expanding window
2. dataset current (train-test-split, 1-fold, train/test/valid has the same y0/y1 ratio)
    * cross section
    * FIM
    * NeuDP batch=256
    * NeuDP all company
    * NeuDP_GAT
        * kmeans
        * industry
    * cross time
    * expanding window


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

## experiment settings
ROOT='/tmp2/cwlin/explainable_credit/codes.credit.relation.dev'
cd $ROOT
ws=12

## training settings
patience=20
batch_size=1
num_layers=2
lstm_num_units=128
dropout_rate=0.25
learning_rate=1e-3
weight_decay=1e-6
max_epoch=300

## training settings
dir=$ROOT/experiments/gru12_index
data_dir=$ROOT/data/
device=$1

python3 $ROOT/codes/main_valid.py 
    --model_dir $dir
    --data_dir $data_dir
    --device $device 
    --all_company_ids_path $data_dir/all_company_ids.csv 
    --num_epochs $max_epoch 
    --patience $patience 
    --batch_size $batch_size 
    --learning_rate $learning_rate 
    --weight_decay $weight_decay 
    --lstm_num_units $lstm_num_units 
    --dropout_rate $dropout_rate 
    --num_layers $num_layers 

num_epochs=$(cat $dir/num_epochs)

python3 $ROOT/codes/main.py 
    --model_dir $dir
    --data_dir $data_dir
    --device $device 
    --all_company_ids_path $data_dir/all_company_ids.csv 
    --num_epochs $max_epoch 
    --patience $patience 
    --batch_size $batch_size 
    --learning_rate $learning_rate 
    --weight_decay $weight_decay 
    --lstm_num_units $lstm_num_units 
    --dropout_rate $dropout_rate 
    --num_layers $num_layers 

python3 $ROOT/codes/report.py
    --pred_file $dir/pred_test.csv 
    --label_file $data_dir/test_cum.csv

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $dir/AR

python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $dir/RMSNE

```