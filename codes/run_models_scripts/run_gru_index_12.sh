#!/bin/bash
echo batch_size=$1
echo date=$2
echo device=$3
echo patience=$4
echo exp_dir_name=$5
echo learning_rate=$6
echo lstm_num_units=$7
echo weight_decay=$8

exp_dir_name=$5

# for i in {01..13};
# do
#     echo fold $i
#     python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/$exp_dir_name/gru01_index/index_fold_${i} --data_dir=./data/8_labels_index/len_01/index_fold_${i} --num_epochs 300 --device=$3 --batch_size $1 --shared_param --patience $4 --learning_rate $6 --lstm_num_units $7 --weight_decay $8
# done
# python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/$exp_dir_name/gru01_index/ --label_dir=./data/8_labels_index/len_01/ --device=$3 --batch_size $1
# python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/$exp_dir_name/gru01_index

# for i in {01..13};
# do
#     echo fold $i
#     python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/$exp_dir_name/gru06_index/index_fold_${i} --data_dir=./data/8_labels_index/len_06/index_fold_${i} --num_epochs 300 --device=$3 --batch_size $1 --shared_param --patience $4 --learning_rate $6 --lstm_num_units $7 --weight_decay $8
# done
# python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/$exp_dir_name/gru06_index/ --label_dir=./data/8_labels_index/len_06/ --device=$3 --batch_size $1
# python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/$exp_dir_name/gru06_index

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/$exp_dir_name/gru12_index/index_fold_${i} --data_dir=./data/8_labels_index/len_12/index_fold_${i} --num_epochs 300 --device=$3 --batch_size $1 --shared_param --patience $4 --learning_rate $6 --lstm_num_units $7 --weight_decay $8
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/$exp_dir_name/gru12_index/ --label_dir=./data/8_labels_index/len_12/ --device=$3 --batch_size $1
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/$exp_dir_name/gru12_index

# python3 get_results.py --exp_dir /tmp2/cwlin/explainable_credit/explainable_credit/$exp_dir_name --exp_model gru --exp_date $2 --output_dir /tmp2/cwlin/explainable_credit/results --exp_desc $exp_desc