date=0312
bs=5
lr=1e-3
device=0
patience=100
lstm_num_units=128
weight_decay=1e-6
num_layers=2
dropout_rate=0.5

hyperparameter_name="all"
exp_dir_name=experiments-tune-$hyperparameter_name
# bash ./explainable_credit/codes/run_models_scripts/run_gru_index_01.sh $bs $date $device $patience $exp_dir_name $lr $lstm_num_units $weight_decay $num_layers $dropout_rate
# mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all
# mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru01_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru01_index/results_pt_01.csv
# cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru01_index/results_pt_01.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all/ws=01.bs=$bs.patience=$patience.hidden=$lstm_num_units.lr=$lr.weight_decay=$weight_decay.${num_layers}layer.dropout=$dropout_rate

# bash ./explainable_credit/codes/run_models_scripts/run_gru_index_06.sh $bs $date $device $patience $exp_dir_name $lr $lstm_num_units $weight_decay $num_layers $dropout_rate
# mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all
# mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru06_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru06_index/results_pt_06.csv
# cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru06_index/results_pt_06.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all/ws=06.bs=$bs.patience=$patience.hidden=$lstm_num_units.lr=$lr.weight_decay=$weight_decay.${num_layers}layer.dropout=$dropout_rate

bash ./explainable_credit/codes/run_models_scripts/run_gru_index_12.sh $bs $date $device $patience $exp_dir_name $lr $lstm_num_units $weight_decay $num_layers $dropout_rate
mkdir /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all
mv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt.csv /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv
cp /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name/gru12_index/results_pt_12.csv /tmp2/cwlin/explainable_credit/results/$date-revised.index.tune.all/ws=12.bs=$bs.patience=$patience.hidden=$lstm_num_units.lr=$lr.weight_decay=$weight_decay.${num_layers}layer.dropout=$dropout_rate