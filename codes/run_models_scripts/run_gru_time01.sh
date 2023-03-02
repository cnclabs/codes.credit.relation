# #!/bin/bash

for i in {01..13};
do
    echo fold $i
    python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru01/time_fold_${i} --data_dir=./data/8_labels_time/len_01/time_fold_${i} --device=0
done
python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru01/ --label_dir=./data/8_labels_time/len_01/ --device=0 --split_type "time"
python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru01

# for i in {01..13};
# do
#     echo fold $i
#     python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru06/time_fold_${i} --data_dir=./data/8_labels_time/len_06/time_fold_${i} --device=0
# done
# python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru06/ --label_dir=./data/8_labels_time/len_06/ --device=0 --split_type "time"
# python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru06

# for i in {01..13};
# do
#     echo fold $i
#     python ./explainable_credit/codes/main.py --model_dir=./explainable_credit/experiments/gru12/time_fold_${i} --data_dir=./data/8_labels_time/len_12/time_fold_${i} --device=0
# done
# python ./explainable_credit/codes/pred_folds.py --exp_dir=./explainable_credit/experiments/gru12/ --label_dir=./data/8_labels_time/len_12/ --device=0 --split_type "time"
# python ./explainable_credit/codes/synthesize_results.py --parent_dir=./explainable_credit/experiments/gru12