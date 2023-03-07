python main.py --model_dir=/tmp2/cwlin/default_prediction/Default_Prediction_Models/experiments/gru06_index/index_fold_06 --data_dir=/tmp2/cwlin/default_prediction/Default_Prediction_Models/data/8_labels_index/len_06/index_fold_06 --device=0
python pred_folds.py --exp_dir=/tmp2/cwlin/default_prediction/Default_Prediction_Models/experiments/gru06_index/ --label_dir=/tmp2/cwlin/default_prediction/Default_Prediction_Models/data/8_labels_index/len_06/
python synthesize_results.py --parent_dir=../../experiments/gru06_index

