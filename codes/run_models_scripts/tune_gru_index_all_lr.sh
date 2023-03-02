date=0210
bs=1
lr=1e-4
device=1
patience=20
lstm_num_units=64
weight_decay=1e-5

# hyperparameter_name='bs'
# bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

# for hyperparameter_value in 1 5 10 15
#     do
#     exp_dir_name=experiments-tune-$hyperparameter_name
#     bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $hyperparameter_value $date $device $patience $exp_dir_name $lr $lstm_num_units $weight_decay
#     mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
#     cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     done

# date=0211
# hyperparameter_name='patience'
# bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

# for hyperparameter_value in 50 75 100
#     do
#     exp_dir_name=experiments-tune-$hyperparameter_name
#     bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $hyperparameter_value $exp_dir_name $lr $lstm_num_units $weight_decay
#     mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
#     cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     done

date=0211
hyperparameter_name='lr'
bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

for hyperparameter_value in 1e-3 1e-5
    do
    exp_dir_name=experiments-tune-$hyperparameter_name
    bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $patience $exp_dir_name $hyperparameter_value $lstm_num_units $weight_decay
    mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
    mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
    cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
    done

# date=0213
# hyperparameter_name='lstm_num_units'
# bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

# for hyperparameter_value in 32 128
#     do
#     exp_dir_name=experiments-tune-$hyperparameter_name
#     bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $patience $exp_dir_name $lr $hyperparameter_value $weight_decay
#     mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
#     cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     done

# date=0213
# hyperparameter_name='weight_decay'
# bash /tmp2/cwlin/explainable_credit/explainable_credit/codes/new_experiments_scripts/new_experiments.sh $hyperparameter_name

# for hyperparameter_value in 1e-4 1e-6
#     do
#     exp_dir_name=experiments-tune-$hyperparameter_name
#     bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $patience $exp_dir_name $lr $lstm_num_units $hyperparameter_value
#     mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
#     cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.$hyperparameter_name=$hyperparameter_value
#     done

