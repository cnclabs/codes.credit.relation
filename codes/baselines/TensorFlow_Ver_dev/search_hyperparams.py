"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Directory containing the model")
parser.add_argument('--split_type', default='index',
                    help="Fold split type")
parser.add_argument('--device', default='0',
                    help="CUDA_DEVICE")


def launch_training_job(parent_dir, data_dir, restore_dir, job_name, device, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py \
            --model_dir {model_dir} \
            --data_dir {data_dir} \
            --device {device} \
            --restore {restore_model}".format(
                    python=PYTHON,
                    model_dir=model_dir,
                    data_dir=data_dir,
                    device=device,
                    restore_model=restore_dir)
    print(cmd)
    check_call(cmd, shell=True)


def run_folds(fold_name, parent_dir, data_dir, restore_dir, split_type, device, params, n_folds=13):

    for fold in range(n_folds):
        # Launch job (name has to be unique)
        job_name = "{}_fold_{:02d}".format(fold_name, fold + 1)
        fold_dir = os.path.join(data_dir, split_type + "_fold_{:02d}".format(fold + 1))
        exp_dir = os.path.join(parent_dir, job_name)
        if os.path.isdir(exp_dir):
            print("{} exist, passed".format(exp_dir))
            continue
        else:
            launch_training_job(parent_dir, fold_dir, restore_dir, job_name, device, params)
        #launch_training_job(parent_dir, fold_dir, restore_dir, job_name, device, params)


def search_hyperparameter(h_parameters,
        parent_dir, data_dir, restore_dir, split_type, device,
        params):
    """TODO: Docstring for search_hyperparameter.

    :arg1: TODO
    :returns: TODO

    """
    for k, v in h_parameters.items():
        for item in v:
            # Modify the relevant parameter in params
            params.dict[k] = item
            fold_name = "{}_{}".format(k, item)
            run_folds(fold_name, parent_dir, data_dir, restore_dir,
                    split_type, device, params)





if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    ## Perform hypersearch over one parameter
    ##h_parameters = {'learning_rate': [1e-4, 1e-3, 1e-2]}
    #h_parameters = {'weight_decay': [1e-5, 1e-3]}
    #search_hyperparameter(h_parameters,
    #        args.parent_dir, args.data_dir, args.restore_dir, args.split_type, args.device, params)
    run_folds(args.split_type, args.parent_dir, args.data_dir, args.restore_dir,
            args.split_type, args.device, params)
