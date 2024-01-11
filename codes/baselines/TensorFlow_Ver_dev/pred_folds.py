"""Peform hyperparemeters search"""

import argparse
import os
import sys
from tqdm import trange
from subprocess import check_call

from model.utils import Params


# PYTHON = sys.executable
PYTHON = '/home/cwlin/miniconda3/envs/default_predict/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='experiments/baseline_fim_insample',
                    help="Directory containing params.json")
parser.add_argument('--data_file', default='test_cum.gz',
                    help="Directory containing params.json")
parser.add_argument('--pred_file', default='pred.csv',
                    help="File name of predictions")
parser.add_argument('--label_dir', default='data/processed',
                    help="Directory containing the dataset")
parser.add_argument('--label_file', default='test_cum.gz',
                    help="Label filename")
parser.add_argument('--split_type', default='index',
                    help="Fold split type")
parser.add_argument('--device', default='cpu',
                    help="CUDA_DEVICE")
## cwlin: predict_folds.py --fold 1
parser.add_argument('--fold', default=13, type=int,
                    help="how many folds to predict")

def predict_folds(exp_dir, data_file, pred_file, label_dir, label_file, split_type, device, num_folds=13):
    """TODO: Docstring for search_hyperparameter.
    """
    for fold in trange(num_folds):
        # Launch job (name has to be unique)
        exp_fold_dir = os.path.join(exp_dir, split_type + "_fold_{:02d}".format(fold + 1))
        label_fold_dir = os.path.join(label_dir, split_type + "_fold_{:02d}".format(fold + 1))

        if pred_file == "pred_test.csv":
            pred_cmd = '{python} ./TensorFlow_Ver_dev/predict.py\
                    --model_dir {exp_fold}\
                    --data_dir {data_dir}\
                    --data_file {data_file}\
                    --restore_dir last_weights\
                    --device {device}'.format(
                            python=PYTHON,
                            exp_fold=exp_fold_dir,
                            data_file=data_file,
                            data_dir=label_fold_dir,
                            device=device)
            check_call(pred_cmd, shell=True)

        score_cmd = '{python} ./TensorFlow_Ver_dev/postprocess/report.py\
                --pred_file {pred_fold}/{pred_file}\
                --label_file {label_fold}/{label_file}'.format(
                        python=PYTHON,
                        pred_fold=exp_fold_dir,
                        pred_file=pred_file,
                        label_fold=label_fold_dir,
                        label_file=label_file
                        )
        check_call(score_cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    #json_path = os.path.join(args.parent_dir, 'params.json')
    #assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    #params = Params(json_path)
    args = parser.parse_args()
    predict_folds(args.exp_dir, args.data_file,
            args.pred_file, args.label_dir, args.label_file, args.split_type, args.device, num_folds=args.fold)
