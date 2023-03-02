date=0210
bs=1
lr=1e-4
device=0
patience=20
lstm_num_units=64
weight_decay=1e-5

date=0211
hyperparameter_name='patience'
bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

for hyperparameter_value in 50 75 100
    do
    exp_dir_name=experiments-tune-$hyperparameter_name
    bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $hyperparameter_value $exp_dir_name $lr $lstm_num_units $weight_decay
    mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
    mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
    cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
    done